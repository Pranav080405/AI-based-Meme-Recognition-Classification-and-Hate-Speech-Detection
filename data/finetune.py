"""
data/finetune.py — Fine-tune CLIP + classifier head on Hateful Memes dataset

Architecture:
  CLIP image encoder + CLIP text encoder
  → concatenate embeddings (512 + 512 = 1024-d)
  → Linear classifier head (1024 → 2)
  → Binary cross-entropy loss

This is the "THIS IS HUGE" step from the upgrade plan.
Even a few epochs on the Hateful Memes dataset will dramatically
improve hate speech detection for meme-specific content.

Usage:
    python -m data.finetune --epochs 5 --batch-size 16 --lr 2e-5
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from data.pipeline import load_dataset_splits
from utils.logger import logger
from config import CLIP_MODEL, RANDOM_SEED


# ─────────────────────────────────────────────
# MODEL: CLIP + CLASSIFIER HEAD
# ─────────────────────────────────────────────

class CLIPHatefulMemesClassifier(nn.Module):
    """
    CLIP backbone with a binary classifier head.
    Fuses image and text embeddings via concatenation.
    """

    def __init__(self, clip_model_name: str = CLIP_MODEL, num_classes: int = 2):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)

        # Freeze most of CLIP — only fine-tune the last 2 transformer layers
        # This prevents catastrophic forgetting while adapting to the domain
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Unfreeze last 2 layers of both encoders
        for name, param in self.clip.named_parameters():
            if any(f"layers.{i}" in name for i in [10, 11]):
                param.requires_grad = True

        embed_dim = self.clip.config.projection_dim  # 512 for ViT-B/32
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        img_emb = self.clip.get_image_features(pixel_values=pixel_values)
        txt_emb = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Normalize embeddings
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        # Fuse and classify
        fused = torch.cat([img_emb, txt_emb], dim=-1)
        return self.classifier(fused)


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return total_loss / len(loader), acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask, pixel_values)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return total_loss / len(loader), acc, f1


# ─────────────────────────────────────────────
# MAIN TRAINING SCRIPT
# ─────────────────────────────────────────────

def main(epochs: int, batch_size: int, lr: float, save_path: str):
    torch.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Fine-tuning on {device}")

    # Data
    train_loader, val_loader, test_loader = load_dataset_splits(batch_size=batch_size)

    # Model
    model = CLIPHatefulMemesClassifier().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    # Loss — weighted for class imbalance (hate speech is rarer)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        logger.info(f"\n{'─'*40}\nEpoch {epoch}/{epochs}")

        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Train — loss: {train_loss:.4f}, acc: {train_acc:.4f}, f1: {train_f1:.4f}\n"
            f"Val   — loss: {val_loss:.4f}, acc: {val_acc:.4f}, f1: {val_f1:.4f}"
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            logger.info(f"✅ Best model saved → {save_path} (val F1={val_f1:.4f})")

    # Final test evaluation
    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    logger.info(
        f"\n{'═'*40}\nFinal Test Results\n"
        f"  Accuracy: {test_acc:.4f}\n"
        f"  F1 Score: {test_f1:.4f}\n"
        f"  Loss: {test_loss:.4f}\n{'═'*40}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on Hateful Memes")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save-path", type=str, default="models/clip_finetuned.pt")
    args = parser.parse_args()

    main(args.epochs, args.batch_size, args.lr, args.save_path)