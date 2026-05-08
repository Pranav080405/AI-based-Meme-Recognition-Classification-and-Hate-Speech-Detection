"""
models/clip_model.py — CLIP multimodal model with fine-tuned hate speech classifier

Loads fine-tuned weights from clip_finetuned.pt if available,
otherwise falls back to zero-shot CLIP analysis.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from utils.logger import logger
from config import CLIP_MODEL, MEME_CATEGORIES

FINETUNED_WEIGHTS = os.path.join(os.path.dirname(__file__), "clip_finetuned.pt")


# ─────────────────────────────────────────────
# FINE-TUNED CLASSIFIER
# ─────────────────────────────────────────────

class CLIPHatefulMemesClassifier(nn.Module):
    def __init__(self, clip_model_name: str = CLIP_MODEL, num_classes: int = 2):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)

        embed_dim = self.clip.config.projection_dim  # 512
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        img_emb = vision_outputs.pooler_output
        img_emb = self.clip.visual_projection(img_emb)

        text_outputs = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        txt_emb = text_outputs.pooler_output
        txt_emb = self.clip.text_projection(txt_emb)

        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        fused = torch.cat([img_emb, txt_emb], dim=-1)
        return self.classifier(fused)


# ─────────────────────────────────────────────
# MAIN CLIP MODEL CLASS
# ─────────────────────────────────────────────

class CLIPMultimodalModel:
    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP processor: {CLIP_MODEL}")
        self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

        # Load fine-tuned classifier if weights exist
        if os.path.exists(FINETUNED_WEIGHTS):
            logger.info(f"Loading fine-tuned weights: {FINETUNED_WEIGHTS}")
            self._classifier = CLIPHatefulMemesClassifier().to(self._device)
            self._classifier.load_state_dict(
                torch.load(FINETUNED_WEIGHTS, map_location=self._device)
            )
            self._classifier.eval()
            self._finetuned = True
            logger.info("Fine-tuned CLIP classifier ready ✅")

            # ── Reuse CLIP backbone — no second load ──
            self._base = self._classifier.clip
            logger.info("Reusing CLIP backbone from fine-tuned model ✅")

        else:
            logger.warning(
                f"No fine-tuned weights found at {FINETUNED_WEIGHTS} "
                f"— falling back to zero-shot CLIP"
            )
            self._classifier = None
            self._finetuned = False

            # Load base CLIP only as fallback
            self._base = CLIPModel.from_pretrained(CLIP_MODEL).to(self._device)
            logger.info(f"Base CLIP model ready on {self._device}")

        self._base.eval()
        logger.info(f"CLIP ready on {self._device}")

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            vision_out = self._base.vision_model(**inputs)
            emb = self._base.visual_projection(vision_out.pooler_output)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().squeeze(0).numpy()

    def get_text_embedding(self, text: str) -> np.ndarray:
        inputs = self._processor(
            text=[text], return_tensors="pt", padding=True
        ).to(self._device)
        with torch.no_grad():
            text_out = self._base.text_model(**inputs)
            emb = self._base.text_projection(text_out.pooler_output)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().squeeze(0).numpy()

    def image_text_similarity(self, image: Image.Image, text: str) -> float:
        img_emb = self.get_image_embedding(image)
        txt_emb = self.get_text_embedding(text)
        similarity = float(np.dot(img_emb, txt_emb))
        logger.debug(f"CLIP similarity: {similarity:.4f}")
        return round(similarity, 4)

    def classify_image(self, image: Image.Image, labels: list = None) -> dict:
        labels = labels or [f"a meme about {c}" for c in MEME_CATEGORIES]
        inputs = self._processor(
            text=labels, images=image,
            return_tensors="pt", padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._base(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1).squeeze(0).cpu().tolist()
        label_scores = {l: round(p, 4) for l, p in zip(labels, probs)}
        top_idx = int(np.argmax(probs))

        return {
            "top_label": labels[top_idx],
            "top_score": round(probs[top_idx], 4),
            "all_scores": label_scores,
        }

    def classify_hate(self, image: Image.Image, text: str) -> dict:
        """
        Run fine-tuned hate speech classifier on image+text.
        Only available if fine-tuned weights are loaded.
        """
        if not self._finetuned or self._classifier is None:
            return {"available": False}

        inputs = self._processor(
            text=[text], images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        ).to(self._device)

        with torch.no_grad():
            logits = self._classifier(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
            )

        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        hate_score = round(probs[1], 4)
        hateful = hate_score >= 0.40

        logger.debug(f"CLIP hate classifier → hate={hate_score:.2f}")

        return {
            "available": True,
            "hateful": hateful,
            "hate_score": hate_score,
            "label": "hateful" if hateful else "non-hateful",
        }

    def analyze(self, image: Image.Image, text: str) -> dict:
        """Full multimodal analysis."""
        try:
            similarity   = self.image_text_similarity(image, text) if text else 0.0
            classification = self.classify_image(image)
            hate         = self.classify_hate(image, text)
            img_emb      = self.get_image_embedding(image)
            txt_emb      = self.get_text_embedding(text) if text else np.zeros(512)

            return {
                "image_text_similarity":  similarity,
                "image_classification":   classification,
                "hate_classification":    hate,
                "image_embedding":        img_emb,
                "text_embedding":         txt_emb,
            }

        except Exception as e:
            logger.error(f"CLIP analysis failed: {e}")
            return {
                "image_text_similarity": 0.0,
                "image_classification": {
                    "top_label": "unknown",
                    "top_score": 0.0,
                    "all_scores": {}
                },
                "hate_classification": {"available": False},
                "image_embedding":  np.zeros(512),
                "text_embedding":   np.zeros(512),
                "error": str(e),
            }