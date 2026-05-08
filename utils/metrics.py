"""
utils/metrics.py — Evaluation metrics for hate speech and sentiment models

Usage (CLI):
    python -m utils.metrics --predictions preds.json --labels labels.json

Predictions JSON format:
    [{"hate": 1, "sentiment": "positive"}, ...]

Labels JSON format:
    [{"hate": 1, "sentiment": "positive"}, ...]
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
from utils.logger import logger


# ─────────────────────────────────────────────
# CORE METRIC FUNCTIONS
# ─────────────────────────────────────────────

def evaluate_binary(y_true: list, y_pred: list, label_name: str = "Task") -> dict:
    """
    Full binary classification report.
    Returns dict of metrics.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty predictions or labels — skipping evaluation")
        return {}

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC only if probabilities differ
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = None

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else "N/A",
    }

    logger.info(f"\n{'─'*40}\n{label_name} Metrics:\n{json.dumps(metrics, indent=2)}")
    print(f"\n[{label_name}] Full Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    return metrics


def evaluate_multiclass(
    y_true: list,
    y_pred: list,
    class_names: list,
    label_name: str = "Task"
) -> dict:
    """
    Multi-class classification report with per-class F1.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }

    print(f"\n[{label_name}] Full Report:\n")
    print(
        classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0
        )
    )
    logger.info(f"{label_name} Metrics: {metrics}")
    return metrics


# ─────────────────────────────────────────────
# CONFUSION MATRIX VISUALIZATION
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: list,
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> None:
    """Plot and optionally save a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info(f"Confusion matrix saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemeIQ Evaluation Metrics")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    parser.add_argument("--labels", required=True, help="Path to ground truth labels JSON")
    args = parser.parse_args()

    with open(args.predictions) as f:
        preds = json.load(f)
    with open(args.labels) as f:
        labels = json.load(f)

    # Hate speech binary eval
    hate_preds = [p["hate"] for p in preds]
    hate_labels = [l["hate"] for l in labels]
    evaluate_binary(hate_labels, hate_preds, label_name="Hate Speech")
    plot_confusion_matrix(
        hate_labels, hate_preds,
        class_names=["Not Hateful", "Hateful"],
        title="Hate Speech Confusion Matrix",
        save_path="logs/hate_confusion.png",
    )

    # Sentiment multi-class eval
    sent_map = {"negative": 0, "neutral": 1, "positive": 2}
    sent_preds = [sent_map.get(p["sentiment"], 1) for p in preds]
    sent_labels = [sent_map.get(l["sentiment"], 1) for l in labels]
    evaluate_multiclass(
        sent_labels, sent_preds,
        class_names=["Negative", "Neutral", "Positive"],
        label_name="Sentiment",
    )
    plot_confusion_matrix(
        sent_labels, sent_preds,
        class_names=["Negative", "Neutral", "Positive"],
        title="Sentiment Confusion Matrix",
        save_path="logs/sentiment_confusion.png",
    )