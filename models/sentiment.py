"""
models/sentiment.py — Sentiment analysis using Twitter-RoBERTa

Model: cardiffnlp/twitter-roberta-base-sentiment
Why: Fine-tuned on 124M tweets. Meme text shares Twitter's informal,
     abbreviation-heavy, emoji-adjacent language style. Outperforms
     raw BERT significantly on this domain.

Labels: LABEL_0 = negative, LABEL_1 = neutral, LABEL_2 = positive
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.text_utils import clean_text, is_empty_text, truncate_for_model
from utils.logger import logger
from config import SENTIMENT_MODEL, SENTIMENT_LABELS, SENTIMENT_CONFIDENCE_THRESHOLD


class SentimentModel:
    """
    Wraps Twitter-RoBERTa for meme text sentiment classification.
    """

    def __init__(self):
        logger.info(f"Loading sentiment model: {SENTIMENT_MODEL}")
        self._tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        self._model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        logger.info(f"Sentiment model ready on {self._device}")

    def predict(self, text: str) -> dict:
        """
        Classify sentiment of meme text.

        Args:
            text: Raw or cleaned text from OCR

        Returns:
            {
              "label": str,         # "positive" | "neutral" | "negative"
              "score": float,       # Confidence 0–1 for winning label
              "scores": dict,       # Scores for all labels
              "low_confidence": bool,
            }
        """
        text = clean_text(text)

        if is_empty_text(text):
            logger.warning("Sentiment: empty text — returning neutral default")
            return {
                "label": "neutral",
                "score": 0.0,
                "scores": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
                "low_confidence": True,
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

            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)
            scores_raw = probs.cpu().tolist()

            # Map raw label IDs to human names
            label_map = SENTIMENT_LABELS  # {"LABEL_0": "negative", ...}
            id_to_name = {i: label_map[f"LABEL_{i}"] for i in range(len(scores_raw))}

            scores = {id_to_name[i]: round(scores_raw[i], 4) for i in range(len(scores_raw))}
            best_id = int(torch.argmax(probs).item())
            best_label = id_to_name[best_id]
            best_score = scores_raw[best_id]

            low_conf = best_score < SENTIMENT_CONFIDENCE_THRESHOLD

            if low_conf:
                logger.warning(
                    f"Sentiment low confidence ({best_score:.2f}) — "
                    f"label='{best_label}' may be unreliable"
                )

            logger.debug(f"Sentiment → {best_label} ({best_score:.2f})")

            return {
                "label": best_label,
                "score": round(best_score, 4),
                "scores": scores,
                "low_confidence": low_conf,
            }

        except Exception as e:
            logger.error(f"Sentiment prediction failed: {e}")
            return {
                "label": "neutral",
                "score": 0.0,
                "scores": {},
                "low_confidence": True,
                "error": str(e),
            }
