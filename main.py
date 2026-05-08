"""
main.py — MemeIQ CLI entry point

Usage:
    python main.py --image path/to/meme.jpg
    python main.py --image meme.png --no-clip
    python main.py --image meme.jpg --json
"""

import argparse
import json
import time
import numpy as np
from PIL import Image

from models.ocr import OCRModel
from models.sentiment import SentimentModel
from models.hate_speech import HateSpeechModel
from models.categorizer import MemeCategorizerModel
from models.clip_model import CLIPMultimodalModel
from utils.image_utils import load_image, validate_image
from utils.logger import logger



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)



_models = {}


def get_models(use_clip: bool = True) -> dict:
    """Load models once and cache them."""
    global _models

    if not _models:
        logger.info("Initializing all models...")

        _models["ocr"] = OCRModel()
        _models["sentiment"] = SentimentModel()
        _models["hate"] = HateSpeechModel()
        _models["categorizer"] = MemeCategorizerModel()

        if use_clip:
            _models["clip"] = CLIPMultimodalModel()

    return _models



def analyze_meme(image: Image.Image, use_clip: bool = True) -> dict:
    """
    Full multimodal meme analysis pipeline.

    Steps:
      1. OCR
      2. Sentiment
      3. Hate speech
      4. CLIP multimodal
      5. Categorization
    """

    start = time.time()

    models = get_models(use_clip=use_clip)


    logger.info("Step 1/5 — OCR")

    ocr_result = models["ocr"].extract(image)
    text = ocr_result["text"]


    logger.info("Step 2/5 — Sentiment")

    sentiment_result = models["sentiment"].predict(text)


    logger.info("Step 3/5 — Hate Speech")

    hate_result = models["hate"].predict(text)


    clip_result = {}

    if use_clip and "clip" in models:
        logger.info("Step 4/5 — CLIP Multimodal")

        clip_analysis = models["clip"].analyze(image, text)

        clip_result = {
            "image_text_similarity": clip_analysis.get(
                "image_text_similarity"
            ),

            "image_classification": clip_analysis.get(
                "image_classification"
            ),

            #  Added hate classification output
            "hate_classification": clip_analysis.get(
                "hate_classification",
                {
                    "available": False,
                    "hate_score": None,
                }
            ),
        }

    else:
        logger.info("Step 4/5 — CLIP skipped")


    logger.info("Step 5/5 — Categorization")

    image_label = (
        clip_result.get("image_classification", {}).get("top_label")
        if clip_result else None
    )

    category_result = models["categorizer"].predict(
        text,
        image_label=image_label
    )


    clip_hate = clip_result.get("hate_classification", {})

    if (
        clip_hate.get("available")
        and clip_hate.get("hate_score") is not None
    ):

        # Average Dehatebert + CLIP fine-tuned scores
        combined_hate_score = round(
            (
                hate_result["hate_score"]
                + clip_hate["hate_score"]
            ) / 2,
            4
        )

        hate_result["hate_score"] = combined_hate_score

        hate_result["hateful"] = (
            combined_hate_score >= 0.55
        )

        hate_result["label"] = (
            "hateful"
            if hate_result["hateful"]
            else "non-hateful"
        )

        hate_result["clip_hate_score"] = (
            clip_hate["hate_score"]
        )

        logger.debug(
            f"Combined hate score: {combined_hate_score:.2f}"
        )


    elapsed = round(time.time() - start, 2)

    logger.info(f"Analysis complete in {elapsed}s")

    return {
        "ocr": ocr_result,
        "sentiment": sentiment_result,
        "hate_speech": hate_result,
        "clip": clip_result,
        "category": category_result,
        "elapsed_seconds": elapsed,
    }



def print_report(result: dict) -> None:
    """Pretty-print the analysis report."""

    sep = "─" * 50

    print(f"\n{'═'*50}")
    print("  🧠  MemeIQ — Analysis Report")
    print(f"{'═'*50}")

  
    ocr = result.get("ocr", {})

    print(f"\n📝  Extracted Text")
    print(sep)

    print(f"  \"{ocr.get('text', 'N/A')}\"")
    print(f"  OCR confidence: {ocr.get('confidence', 0):.0%}")

  
    sent = result.get("sentiment", {})

    emoji = {
        "positive": "😊",
        "neutral": "😐",
        "negative": "😠",
    }.get(sent.get("label"), "🤔")

    print(f"\n{emoji}  Sentiment")
    print(sep)

    print(
        f"  Label: {sent.get('label', 'N/A').upper()}  "
        f"({sent.get('score', 0):.0%} confident)"
    )

    if sent.get("low_confidence"):
        print("  ⚠️  Low confidence — result may be unreliable")

 
    hate = result.get("hate_speech", {})

    flag = (
        "🚨"
        if hate.get("hateful")
        else (
            "⚠️"
            if hate.get("needs_review")
            else "✅"
        )
    )

    print(f"\n{flag}  Hate Speech Detection")
    print(sep)

    print(f"  Label: {hate.get('label', 'N/A').upper()}")
    print(f"  Hate score: {hate.get('hate_score', 0):.0%}")

    if hate.get("clip_hate_score") is not None:
        print(
            f"  CLIP hate score: "
            f"{hate.get('clip_hate_score', 0):.0%}"
        )

    if hate.get("needs_review"):
        print(
            "  ⚠️  Flagged for human review "
            "(borderline confidence)"
        )


    clip = result.get("clip", {})

    if clip:
        print(f"\n🎯  CLIP Multimodal Analysis")
        print(sep)

        img_cls = clip.get("image_classification", {})

        print(
            f"  Image class: "
            f"{img_cls.get('top_label', 'N/A')} "
            f"({img_cls.get('top_score', 0):.0%})"
        )

        print(
            f"  Image↔Text similarity: "
            f"{clip.get('image_text_similarity', 0):.2f}"
        )

    
    cat = result.get("category", {})

    print(f"\n🏷️   Meme Category")
    print(sep)

    print(
        f"  Category: {cat.get('category', 'N/A')} "
        f"({cat.get('score', 0):.0%} confident)"
    )

    if cat.get("low_confidence"):
        print("  ⚠️  Low confidence categorization")


    print(
        f"\n⏱️  Completed in "
        f"{result.get('elapsed_seconds', '?')}s"
    )

    print(f"{'═'*50}\n")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="MemeIQ — AI Meme Analyzer"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to meme image"
    )

    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Skip CLIP (faster)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON only"
    )

    args = parser.parse_args()

    try:
        image = load_image(args.image)
        validate_image(image)

    except ValueError as e:
        print(f"❌ Image error: {e}")
        exit(1)

    result = analyze_meme(
        image,
        use_clip=not args.no_clip
    )

    if args.json:
        print(
            json.dumps(
                result,
                indent=2,
                cls=NumpyEncoder
            )
        )

    else:
        print_report(result)

        print(
            json.dumps(
                result,
                indent=2,
                cls=NumpyEncoder
            )
        )