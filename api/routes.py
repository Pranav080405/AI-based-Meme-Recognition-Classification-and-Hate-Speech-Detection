"""
api/routes.py — FastAPI route definitions for MemeIQ
"""

import io
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from PIL import Image

from main import analyze_meme, get_models
from utils.image_utils import load_image, validate_image
from utils.logger import logger
from api.schemas import AnalysisResponse, HealthResponse, ErrorResponse

router = APIRouter()


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model load status."""
    try:
        models = get_models(use_clip=True)
        loaded = len(models) > 0
    except Exception:
        loaded = False
    return HealthResponse(status="ok", models_loaded=loaded)


# ─────────────────────────────────────────────
# FULL ANALYSIS
# ─────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Full multimodal meme analysis",
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def analyze(
    file: UploadFile = File(..., description="Meme image (JPEG, PNG, GIF, WEBP)"),
    use_clip: bool = Query(True, description="Enable CLIP multimodal analysis"),
):
    """
    Upload a meme image and receive:
    - Extracted text (OCR)
    - Sentiment (positive/neutral/negative)
    - Hate speech detection
    - CLIP image-text similarity + image classification
    - Zero-shot meme category
    """
    logger.info(f"POST /analyze — file='{file.filename}', use_clip={use_clip}")

    # Validate file type
    allowed = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {allowed}",
        )

    # Load and validate image
    try:
        raw_bytes = await file.read()
        image = load_image(raw_bytes)
        validate_image(image)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Image load failed: {e}")
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    # Run analysis
    try:
        result = analyze_meme(image, use_clip=use_clip)
    except Exception as e:
        logger.error(f"Analysis pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return result


# ─────────────────────────────────────────────
# HATE SPEECH ONLY
# ─────────────────────────────────────────────

@router.post("/hate-speech", tags=["Analysis"], summary="Hate speech detection only")
async def hate_speech_only(
    file: UploadFile = File(...),
):
    """Lighter endpoint — OCR + hate speech detection only. Faster than /analyze."""
    logger.info(f"POST /hate-speech — file='{file.filename}'")
    try:
        raw_bytes = await file.read()
        image = load_image(raw_bytes)
        validate_image(image)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    models = get_models(use_clip=False)
    ocr = models["ocr"].extract(image)
    hate = models["hate"].predict(ocr["text"])

    return {
        "text": ocr["text"],
        "hate_speech": hate,
    }


# ─────────────────────────────────────────────
# SENTIMENT ONLY
# ─────────────────────────────────────────────

@router.post("/sentiment", tags=["Analysis"], summary="Sentiment analysis only")
async def sentiment_only(
    file: UploadFile = File(...),
):
    """OCR + sentiment only. Faster than /analyze."""
    logger.info(f"POST /sentiment — file='{file.filename}'")
    try:
        raw_bytes = await file.read()
        image = load_image(raw_bytes)
        validate_image(image)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    models = get_models(use_clip=False)
    ocr = models["ocr"].extract(image)
    sentiment = models["sentiment"].predict(ocr["text"])

    return {
        "text": ocr["text"],
        "sentiment": sentiment,
    }