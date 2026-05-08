"""
models/hate_speech.py — Hate speech detection using Dehatebert

Model: Hate-speech-CNERG/dehatebert-mono-english
Why: Purpose-built BERT fine-tuned on multiple hate speech datasets.
     Outperforms keyword lists dramatically — understands context,
     coded language, and implicit bias.

Labels: hateful | non-hateful
Confidence threshold: if score < HATE_SPEECH_THRESHOLD → flag for human review
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.text_utils import clean_text, is_empty_text, truncate_for_model
from utils.logger import logger
from config import HATE_SPEECH_MODEL, HATE_SPEECH_THRESHOLD


class HateSpeechModel:
    """
    Wraps Dehatebert for binary hate speech detection.
    """

    def __init__(self):
        logger.info(f"Loading hate speech model: {HATE_SPEECH_MODEL}")
        self._tokenizer = AutoTokenizer.from_pretrained(HATE_SPEECH_MODEL)
        self._model = AutoModelForSequenceClassification.from_pretrained(HATE_SPEECH_MODEL)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        logger.info(f"Hate speech model ready on {self._device}")

    def predict(self, text: str) -> dict:
        """
        Classify text as hateful or non-hateful.

        Args:
            text: Meme text (raw or pre-cleaned)

        Returns:
            {
              "hateful": bool,
              "label": str,           # "hateful" | "non-hateful"
              "score": float,         # Confidence for predicted label
              "hate_score": float,    # Raw probability of being hateful
              "needs_review": bool,   # True if confidence < threshold
            }
        """
        text = clean_text(text)

        if is_empty_text(text):
            logger.warning("Hate speech: empty text — defaulting to non-hateful")
            return {
                "hateful": False,
                "label": "non-hateful",
                "score": 1.0,
                "hate_score": 0.0,
                "needs_review": False,
            }

        text = truncate_for_model(text, max_tokens=512)

        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**inputs).logits

            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

            # Dehatebert: label 0 = hate, label 1 = non-hate (verify per model card)
            # Mapping is confirmed from the model card
            hate_score = round(probs[0], 4)
            non_hate_score = round(probs[1], 4)

            hateful = hate_score >= HATE_SPEECH_THRESHOLD
            winning_score = hate_score if hateful else non_hate_score
            needs_review = (
                hate_score >= 0.40 and hate_score < HATE_SPEECH_THRESHOLD
            )

            logger.debug(
                f"Hate speech → {'HATEFUL' if hateful else 'clean'} "
                f"(hate={hate_score:.2f}, review={needs_review})"
            )

            if needs_review:
                logger.warning(
                    f"Content flagged for human review — hate score {hate_score:.2f} "
                    f"is below threshold {HATE_SPEECH_THRESHOLD}"
                )

            return {
                "hateful": hateful,
                "label": "hateful" if hateful else "non-hateful",
                "score": round(winning_score, 4),
                "hate_score": hate_score,
                "needs_review": needs_review,
            }

        except Exception as e:
            logger.error(f"Hate speech prediction failed: {e}")
            return {
                "hateful": False,
                "label": "non-hateful",
                "score": 0.0,
                "hate_score": 0.0,
                "needs_review": True,
                "error": str(e),
            }
