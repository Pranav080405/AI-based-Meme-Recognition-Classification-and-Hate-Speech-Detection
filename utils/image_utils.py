"""
utils/image_utils.py — Image loading, validation, and preprocessing
"""

import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from utils.logger import logger


MAX_IMAGE_SIZE_MB = 10


def load_image(source) -> Image.Image:
    """
    Load a PIL Image from a file path, bytes, or file-like object.
    Returns RGB PIL Image.
    """
    try:
        if isinstance(source, (str, bytes, bytearray)):
            if isinstance(source, str):
                img = Image.open(source)
            else:
                img = Image.open(io.BytesIO(source))
        elif isinstance(source, Image.Image):
            img = source
        else:
            img = Image.open(source)

        img = img.convert("RGB")
        logger.debug(f"Image loaded — size: {img.size}, mode: {img.mode}")
        return img

    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise ValueError(f"Could not load image: {e}") from e


def validate_image(image: Image.Image, max_mb: float = MAX_IMAGE_SIZE_MB) -> None:
    """Raise ValueError if image fails basic checks."""
    if image is None:
        raise ValueError("Image is None")

    w, h = image.size
    if w < 10 or h < 10:
        raise ValueError(f"Image too small: {w}x{h}")

    # Approximate memory check (RGB = 3 bytes/pixel)
    approx_mb = (w * h * 3) / (1024 ** 2)
    if approx_mb > max_mb * 10:   # generous upper bound
        raise ValueError(f"Image too large: ~{approx_mb:.1f} MB")

    logger.debug(f"Image validated — {w}x{h}")


def enhance_for_ocr(image: Image.Image) -> Image.Image:
    """
    Enhance contrast and sharpness to improve OCR accuracy on meme fonts.
    EasyOCR struggles with low-contrast text — this helps significantly.
    """
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.8)

    # Sharpen
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)

    # Convert to grayscale for OCR, then back to RGB so pipeline stays consistent
    gray = image.convert("L")

    # Apply adaptive threshold via OpenCV for meme text
    gray_np = np.array(gray)
    thresh = cv2.adaptiveThreshold(
        gray_np, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    enhanced = Image.fromarray(thresh).convert("RGB")
    logger.debug("Image enhanced for OCR")
    return enhanced


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL Image (RGB) → OpenCV array (BGR)."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(array: np.ndarray) -> Image.Image:
    """Convert OpenCV array (BGR) → PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))