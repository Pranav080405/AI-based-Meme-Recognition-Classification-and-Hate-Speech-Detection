"""
data/pipeline.py — Dataset loading and preprocessing pipeline

Loads the Hateful Memes dataset from HuggingFace and prepares it
for training or evaluation. Handles image download, OCR, and batching.

Dataset: limjiayi/hateful_memes_expanded
  - label: 0 = non-hateful, 1 = hateful
  - img: image path
  - text: caption text (provided — no OCR needed for training)

Usage:
    from data.pipeline import MemeDataset, load_dataset_splits
    train, val, test = load_dataset_splits()
"""

import os
import io
import torch
import requests
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import CLIPProcessor
from utils.logger import logger
from config import (
    DATASET_NAME, CLIP_MODEL,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED
)


# ─────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────

class MemeDataset(Dataset):
    """
    PyTorch Dataset for the Hateful Memes dataset.
    Each item returns CLIP-processed image + text tensors + label.
    """

    def __init__(self, hf_dataset, processor: CLIPProcessor):
        self.data = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Image — dataset provides PIL images directly
        image = item.get("image")
        if image is None:
            # Fallback: white placeholder
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))

        text = item.get("text", "")
        label = int(item.get("label", 0))

        # CLIP preprocessing
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "text": text,
        }


# ─────────────────────────────────────────────
# DATASET LOADING & SPLITTING
# ─────────────────────────────────────────────

def load_raw_dataset():
    """
    Download and return the raw HuggingFace dataset.
    Cached locally after first download.
    """
    logger.info(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)
    logger.info(f"Dataset loaded — splits: {list(ds.keys())}")
    return ds


def load_dataset_splits(batch_size: int = 16, num_workers: int = 2):
    """
    Load dataset, apply CLIP preprocessing, and return DataLoaders
    for train, val, and test splits.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from transformers import CLIPProcessor

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    ds = load_raw_dataset()

    # Use provided train/test splits; create val from train
    if "train" in ds and "test" in ds:
        raw_train = ds["train"]
        raw_test = ds["test"]

        # Split train → train + val
        val_size = int(len(raw_train) * VAL_SPLIT)
        train_val = raw_train.train_test_split(
            test_size=val_size,
            seed=RANDOM_SEED,
        )
        raw_train = train_val["train"]
        raw_val = train_val["test"]
    else:
        # If only one split exists, do a 3-way split manually
        full = ds[list(ds.keys())[0]]
        split1 = full.train_test_split(test_size=(1 - TRAIN_SPLIT), seed=RANDOM_SEED)
        raw_train = split1["train"]
        remaining = split1["test"]
        val_frac = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
        split2 = remaining.train_test_split(test_size=(1 - val_frac), seed=RANDOM_SEED)
        raw_val = split2["train"]
        raw_test = split2["test"]

    logger.info(
        f"Split sizes — train: {len(raw_train)}, "
        f"val: {len(raw_val)}, test: {len(raw_test)}"
    )

    train_ds = MemeDataset(raw_train, processor)
    val_ds = MemeDataset(raw_val, processor)
    test_ds = MemeDataset(raw_test, processor)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# QUICK PREVIEW
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train, val, test = load_dataset_splits(batch_size=4)
    batch = next(iter(train))
    print("Batch keys:", list(batch.keys()))
    print("pixel_values shape:", batch["pixel_values"].shape)
    print("input_ids shape:", batch["input_ids"].shape)
    print("Labels:", batch["label"].tolist())
    print("Texts:", batch["text"])