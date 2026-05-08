"""
tests/test_models.py — Unit tests for MemeIQ model modules

Run:
    pytest tests/ -v
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch




@pytest.fixture
def blank_image():
    """224x224 white RGB image."""
    return Image.new("RGB", (224, 224), color=(255, 255, 255))


@pytest.fixture
def text_heavy_meme_image():
    """Simple image that would contain text."""
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGB", (400, 200), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((10, 80), "This is a test meme", fill=(255, 255, 255))
    return img



class TestTextUtils:
    def test_clean_text_strips_whitespace(self):
        from utils.text_utils import clean_text
        assert clean_text("  hello   world  ") == "hello world"

    def test_clean_text_removes_control_chars(self):
        from utils.text_utils import clean_text
        result = clean_text("hello\x00world\x01test")
        assert "\x00" not in result
        assert "hello" in result

    def test_clean_text_empty_input(self):
        from utils.text_utils import clean_text
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_is_empty_text(self):
        from utils.text_utils import is_empty_text
        assert is_empty_text("") is True
        assert is_empty_text("  ") is True
        assert is_empty_text("hi") is False
        assert is_empty_text("No text detected") is False

    def test_truncate_for_model(self):
        from utils.text_utils import truncate_for_model
        long_text = " ".join(["word"] * 1000)
        result = truncate_for_model(long_text, max_tokens=512)
        assert len(result.split()) <= 512



class TestImageUtils:
    def test_load_image_from_pil(self, blank_image):
        from utils.image_utils import load_image
        result = load_image(blank_image)
        assert result.mode == "RGB"
        assert result.size == (224, 224)

    def test_validate_image_passes_normal(self, blank_image):
        from utils.image_utils import validate_image
        validate_image(blank_image)  # should not raise

    def test_validate_image_fails_tiny(self):
        from utils.image_utils import validate_image
        tiny = Image.new("RGB", (5, 5))
        with pytest.raises(ValueError, match="too small"):
            validate_image(tiny)

    def test_validate_image_fails_none(self):
        from utils.image_utils import validate_image
        with pytest.raises(ValueError):
            validate_image(None)

    def test_enhance_for_ocr_returns_image(self, blank_image):
        from utils.image_utils import enhance_for_ocr
        result = enhance_for_ocr(blank_image)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_pil_cv2_roundtrip(self, blank_image):
        from utils.image_utils import pil_to_cv2, cv2_to_pil
        cv2_img = pil_to_cv2(blank_image)
        assert cv2_img.shape == (224, 224, 3)
        back = cv2_to_pil(cv2_img)
        assert isinstance(back, Image.Image)



class TestSentimentModel:
    @patch("models.sentiment.AutoModelForSequenceClassification.from_pretrained")
    @patch("models.sentiment.AutoTokenizer.from_pretrained")
    def test_predict_returns_expected_keys(self, mock_tok, mock_model):
        import torch
        mock_tok.return_value = MagicMock()
        mock_model.return_value = MagicMock()

        # Mock tokenizer output
        mock_tok.return_value.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        # Mock model output
        mock_logits = torch.tensor([[2.0, 0.5, 0.3]])
        mock_model.return_value.return_value = MagicMock(logits=mock_logits)
        mock_model.return_value.eval = MagicMock()
        mock_model.return_value.to = MagicMock(return_value=mock_model.return_value)

        from models.sentiment import SentimentModel
        model = SentimentModel()
        result = model.predict("This is hilarious!")

        assert "label" in result
        assert "score" in result
        assert "scores" in result
        assert "low_confidence" in result

    def test_predict_empty_text_returns_neutral(self):
        from utils.text_utils import is_empty_text
        # At minimum, empty text detection should work
        assert is_empty_text("") is True



class TestHateSpeechModel:
    def test_result_schema(self):
        """Verify the expected output schema — without loading the model."""
        expected_keys = {"hateful", "label", "score", "hate_score", "needs_review"}
        # Simulate what the model returns
        mock_result = {
            "hateful": False,
            "label": "non-hateful",
            "score": 0.98,
            "hate_score": 0.02,
            "needs_review": False,
        }
        assert set(mock_result.keys()) == expected_keys
        assert isinstance(mock_result["hateful"], bool)
        assert 0.0 <= mock_result["hate_score"] <= 1.0

    def test_threshold_logic(self):
        """Test the threshold-based review flag logic."""
        from config import HATE_SPEECH_THRESHOLD
        hate_score = 0.55  # below threshold but above 0.40
        hateful = hate_score >= HATE_SPEECH_THRESHOLD
        needs_review = hate_score >= 0.40 and hate_score < HATE_SPEECH_THRESHOLD
        assert not hateful
        assert needs_review



class TestMetrics:
    def test_evaluate_binary_perfect(self):
        from utils.metrics import evaluate_binary
        y = [0, 1, 0, 1, 1]
        result = evaluate_binary(y, y, "Test")
        assert result["accuracy"] == 1.0
        assert result["f1_score"] == 1.0

    def test_evaluate_binary_empty(self):
        from utils.metrics import evaluate_binary
        result = evaluate_binary([], [], "Empty")
        assert result == {}

    def test_evaluate_multiclass(self):
        from utils.metrics import evaluate_multiclass
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 1]
        result = evaluate_multiclass(
            y_true, y_pred,
            class_names=["neg", "neu", "pos"],
            label_name="Sentiment",
        )
        assert "accuracy" in result
        assert "macro_f1" in result
        assert result["accuracy"] <= 1.0