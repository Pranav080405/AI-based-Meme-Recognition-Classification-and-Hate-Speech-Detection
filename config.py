"""
config.py — Central configuration for MemeIQ
All model names, thresholds, and constants live here.
"""

from dataclasses import dataclass, field
from typing import List



SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
HATE_SPEECH_MODEL = "Hate-speech-CNERG/dehatebert-mono-english"
ZERO_SHOT_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
CLIP_MODEL = "openai/clip-vit-base-patch32"



MEME_CATEGORIES: List[str] = [
    "pop culture and entertainment",
    "anime and manga",
    "gaming",
    "sports",
    "political commentary",
    "dark humor",
    "motivational",
    "relationship humor",
    "absurdist or surreal",
    "historical reference",
]


SENTIMENT_LABELS = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


HATE_SPEECH_THRESHOLD = 0.70        # below → flag for human review
CATEGORY_CONFIDENCE_THRESHOLD = 0.40  # below → "Uncategorized"
SENTIMENT_CONFIDENCE_THRESHOLD = 0.50


IMAGE_SIZE = 224
CLIP_IMAGE_SIZE = 224


LOG_DIR = "logs"
LOG_FILE = "logs/memeiq.log"
LOG_LEVEL = "DEBUG"



API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_IMAGE_SIZE_MB = 10



DATASET_NAME = "limjiayi/hateful_memes_expanded"   # HuggingFace dataset
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42