"""
models/categorizer.py — Zero-shot meme categorization (lightweight)
Uses a smaller NLI model instead of BART-large to avoid memory crashes on CPU.
"""

from transformers import pipeline
from utils.text_utils import clean_text, is_empty_text, truncate_for_model
from utils.logger import logger
from config import MEME_CATEGORIES, CATEGORY_CONFIDENCE_THRESHOLD

ZERO_SHOT_MODEL = "typeform/distilbert-base-uncased-mnli"

class MemeCategorizerModel:
    def __init__(self, categories: list = None):
        self._categories = categories or MEME_CATEGORIES
        logger.info(f"Loading zero-shot classifier: {ZERO_SHOT_MODEL}")
        self._pipe = pipeline(
            "zero-shot-classification",
            model=ZERO_SHOT_MODEL,
            device=-1,
        )
        logger.info(f"Categorizer ready — {len(self._categories)} categories")

    def predict(self, text: str, image_label: str = None) -> dict:
        text = clean_text(text)
        if image_label:
            combined = f"{text} [Image contains: {image_label}]"
        else:
            combined = text

        if is_empty_text(combined.replace(image_label or "", "")):
            return self._uncategorized()

        combined = truncate_for_model(combined, max_tokens=400)

        try:
            result = self._pipe(
                combined,
                candidate_labels=self._categories,
                multi_label=False,
            )
            labels = result["labels"]
            scores = result["scores"]
            top_label = labels[0]
            top_score = round(scores[0], 4)
            all_scores = {l: round(s, 4) for l, s in zip(labels, scores)}
            low_conf = top_score < CATEGORY_CONFIDENCE_THRESHOLD

            logger.debug(f"Category → '{top_label}' ({top_score:.2f})")

            return {
                "category": "Uncategorized" if low_conf else top_label,
                "score": top_score,
                "all_scores": all_scores,
                "low_confidence": low_conf,
                "uncategorized": low_conf,
            }
        except Exception as e:
            logger.error(f"Categorization failed: {e}")
            return self._uncategorized(error=str(e))

    def _uncategorized(self, error: str = None) -> dict:
        result = {
            "category": "Uncategorized",
            "score": 0.0,
            "all_scores": {},
            "low_confidence": True,
            "uncategorized": True,
        }
        if error:
            result["error"] = error
        return result