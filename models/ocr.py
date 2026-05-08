"""
models/ocr.py — Text extraction from meme images using EasyOCR
with contrast-enhanced preprocessing for better accuracy on meme fonts.
"""

import numpy as np
import easyocr
from PIL import Image

from utils.image_utils import enhance_for_ocr, pil_to_cv2
from utils.text_utils import clean_text, is_empty_text
from utils.logger import logger


class OCRModel:
    """
    Wraps EasyOCR with meme-specific preprocessing.
    Singleton-style: instantiate once, call extract() per image.
    """

    def __init__(self, languages: list = None):
        languages = languages or ["en"]
        logger.info(f"Loading EasyOCR for languages: {languages}")
        self._reader = easyocr.Reader(languages, gpu=self._gpu_available())
        logger.info("EasyOCR ready")

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def extract(self, image: Image.Image, enhance: bool = True) -> dict:
        """
        Extract text from a PIL image.

        Args:
            image: RGB PIL Image
            enhance: Whether to apply contrast/sharpness enhancement first

        Returns:
            {
              "text": str,          # Full extracted text
              "confidence": float,  # Mean confidence across detected regions
              "regions": list,      # Raw EasyOCR results
              "empty": bool,        # True if no meaningful text found
            }
        """
        try:
            source = enhance_for_ocr(image) if enhance else image
            image_np = pil_to_cv2(source)
            results = self._reader.readtext(image_np)

            if not results:
                logger.warning("EasyOCR returned no text regions")
                return {"text": "", "confidence": 0.0, "regions": [], "empty": True}

            texts = [r[1] for r in results]
            confidences = [r[2] for r in results]

            raw_text = " ".join(texts)
            text = clean_text(raw_text)
            mean_conf = round(float(np.mean(confidences)), 4)

            logger.debug(
                f"OCR extracted {len(results)} regions | "
                f"conf={mean_conf:.2f} | text='{text[:60]}...'"
            )

            return {
                "text": text,
                "confidence": mean_conf,
                "regions": results,
                "empty": is_empty_text(text),
            }

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {"text": "", "confidence": 0.0, "regions": [], "empty": True, "error": str(e)}