"""
api/schemas.py — Pydantic request/response models for MemeIQ API
"""

from typing import Optional, Dict
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# RESPONSE COMPONENTS
# ─────────────────────────────────────────────

class OCRResult(BaseModel):
    text: str
    confidence: float
    empty: bool


class SentimentResult(BaseModel):
    label: str
    score: float
    scores: Dict[str, float]
    low_confidence: bool


class HateSpeechResult(BaseModel):
    hateful: bool
    label: str
    score: float
    hate_score: float
    needs_review: bool


class ImageClassification(BaseModel):
    top_label: str
    top_score: float
    all_scores: Dict[str, float]


class CLIPResult(BaseModel):
    image_text_similarity: float
    image_classification: ImageClassification


class CategoryResult(BaseModel):
    category: str
    score: float
    all_scores: Dict[str, float]
    low_confidence: bool
    uncategorized: bool


# ─────────────────────────────────────────────
# FULL ANALYSIS RESPONSE
# ─────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    ocr: OCRResult
    sentiment: SentimentResult
    hate_speech: HateSpeechResult
    clip: Optional[CLIPResult] = None
    category: CategoryResult
    elapsed_seconds: float

    class Config:
        json_schema_extra = {
            "example": {
                "ocr": {
                    "text": "When you debug for 6 hours and it was a missing semicolon",
                    "confidence": 0.91,
                    "empty": False,
                },
                "sentiment": {
                    "label": "negative",
                    "score": 0.72,
                    "scores": {"negative": 0.72, "neutral": 0.20, "positive": 0.08},
                    "low_confidence": False,
                },
                "hate_speech": {
                    "hateful": False,
                    "label": "non-hateful",
                    "score": 0.98,
                    "hate_score": 0.02,
                    "needs_review": False,
                },
                "clip": {
                    "image_text_similarity": 0.31,
                    "image_classification": {
                        "top_label": "a meme about gaming",
                        "top_score": 0.42,
                        "all_scores": {},
                    },
                },
                "category": {
                    "category": "gaming",
                    "score": 0.62,
                    "all_scores": {},
                    "low_confidence": False,
                    "uncategorized": False,
                },
                "elapsed_seconds": 2.41,
            }
        }


class HealthResponse(BaseModel):
    status: str = "ok"
    models_loaded: bool
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None