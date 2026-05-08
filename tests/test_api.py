"""
tests/test_api.py — Integration tests for MemeIQ FastAPI endpoints

Run:
    pytest tests/test_api.py -v
"""

import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image



@pytest.fixture
def client():
    """Create FastAPI test client with mocked model loading."""
    mock_result = {
        "ocr": {"text": "test meme text", "confidence": 0.9, "empty": False, "regions": []},
        "sentiment": {
            "label": "positive", "score": 0.82,
            "scores": {"positive": 0.82, "neutral": 0.12, "negative": 0.06},
            "low_confidence": False,
        },
        "hate_speech": {
            "hateful": False, "label": "non-hateful",
            "score": 0.97, "hate_score": 0.03, "needs_review": False,
        },
        "clip": {
            "image_text_similarity": 0.31,
            "image_classification": {
                "top_label": "a meme about gaming",
                "top_score": 0.45, "all_scores": {},
            },
        },
        "category": {
            "category": "gaming", "score": 0.61,
            "all_scores": {}, "low_confidence": False, "uncategorized": False,
        },
        "elapsed_seconds": 1.23,
    }

    with patch("main.get_models", return_value=MagicMock()):
        with patch("main.analyze_meme", return_value=mock_result):
            from api.app import app
            return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create a minimal JPEG image as bytes."""
    img = Image.new("RGB", (100, 100), color=(128, 200, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()



class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "models_loaded" in data
        assert data["status"] == "ok"



class TestAnalyzeEndpoint:
    def test_analyze_returns_200(self, client, sample_image_bytes):
        response = client.post(
            "/analyze",
            files={"file": ("meme.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_analyze_response_has_required_fields(self, client, sample_image_bytes):
        data = client.post(
            "/analyze",
            files={"file": ("meme.jpg", sample_image_bytes, "image/jpeg")},
        ).json()
        for key in ["ocr", "sentiment", "hate_speech", "category", "elapsed_seconds"]:
            assert key in data, f"Missing key: {key}"

    def test_analyze_rejects_unsupported_type(self, client):
        response = client.post(
            "/analyze",
            files={"file": ("doc.pdf", b"fake content", "application/pdf")},
        )
        assert response.status_code == 400

    def test_analyze_without_clip(self, client, sample_image_bytes):
        response = client.post(
            "/analyze?use_clip=false",
            files={"file": ("meme.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_analyze_missing_file_returns_422(self, client):
        response = client.post("/analyze")
        assert response.status_code == 422



class TestHateSpeechEndpoint:
    def test_returns_200(self, client, sample_image_bytes):
        with patch("api.routes.get_models") as mock_models:
            mock_ocr = MagicMock()
            mock_ocr.extract.return_value = {"text": "hello", "confidence": 0.9, "empty": False, "regions": []}
            mock_hate = MagicMock()
            mock_hate.predict.return_value = {
                "hateful": False, "label": "non-hateful",
                "score": 0.95, "hate_score": 0.05, "needs_review": False,
            }
            mock_models.return_value = {"ocr": mock_ocr, "hate": mock_hate}

            response = client.post(
                "/hate-speech",
                files={"file": ("meme.jpg", sample_image_bytes, "image/jpeg")},
            )
            assert response.status_code == 200

    def test_response_contains_text_and_hate(self, client, sample_image_bytes):
        with patch("api.routes.get_models") as mock_models:
            mock_ocr = MagicMock()
            mock_ocr.extract.return_value = {"text": "test", "confidence": 0.9, "empty": False, "regions": []}
            mock_hate = MagicMock()
            mock_hate.predict.return_value = {
                "hateful": False, "label": "non-hateful",
                "score": 0.95, "hate_score": 0.05, "needs_review": False,
            }
            mock_models.return_value = {"ocr": mock_ocr, "hate": mock_hate}

            data = client.post(
                "/hate-speech",
                files={"file": ("meme.jpg", sample_image_bytes, "image/jpeg")},
            ).json()

            assert "text" in data
            assert "hate_speech" in data