"""
utils/text_utils.py — Text cleaning and validation helpers
"""

import re
from utils.logger import logger


def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    logger.debug(f"Cleaned text ({len(text)} chars): {text[:80]}...")
    return text


def is_empty_text(text: str) -> bool:
    if not text:
        return True
    stripped = re.sub(r"\s+", "", text)
    return len(stripped) < 3


def truncate_for_model(text: str, max_tokens: int = 512) -> str:
    max_words = int(max_tokens * 0.75)
    words = text.split()
    if len(words) > max_words:
        logger.warning(f"Text truncated from {len(words)} words to {max_words}")
        return " ".join(words[:max_words])
    return text