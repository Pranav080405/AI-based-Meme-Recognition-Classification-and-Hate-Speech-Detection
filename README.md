#  MemeIQ — AI-Powered Meme Analysis & Hate Speech Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade multimodal AI system for meme understanding, hate speech detection, sentiment analysis, and category classification.**

*Research paper shortlisted for publication — authored with Rohan Madanu under the guidance of Dr. Mohan.*

[Features](#-features) • [Architecture](#-architecture) • [Models](#-models-used) • [Setup](#-quick-start) • [API](#-api-reference) • [Results](#-results)

</div>

---

##  What is MemeIQ?

Memes are one of the most complex forms of digital communication — they blend image, text, cultural context, and tone in ways that standard NLP tools completely fail to understand. A keyword-based hate speech detector will flag *"When you destroy them in chess"* as violent. A text-only sentiment model has no idea that a smiling Tom & Jerry image changes the meaning of everything.

MemeIQ solves this with a **true multimodal pipeline** — combining OCR, transformer-based NLP, and CLIP's joint image-text embedding space to analyze memes the way humans actually read them.

---

##  Features

- ** OCR with preprocessing** — EasyOCR with contrast enhancement and adaptive thresholding for meme fonts
- ** Domain-specific sentiment** — Twitter-RoBERTa trained on 124M tweets, matched to meme language style
- ** Contextual hate detection** — Dehatebert + fine-tuned CLIP ensemble with confidence thresholds and human review flags
- ** True multimodality** — CLIP ViT-B/32 encodes image and text into the same embedding space for genuine cross-modal understanding
- ** Zero-shot categorization** — BART-MNLI classifies memes into 10 categories without any training
- ** FastAPI backend** — async REST API with Pydantic validation and Swagger docs
- ** Streamlit UI** — dark-themed interactive interface with score visualizations
- ** Evaluation metrics** — F1, precision, recall, confusion matrix, ROC-AUC
- ** Fine-tuned on Hateful Memes** — CLIP classifier trained on Facebook's 10k hateful memes dataset

---

##  Architecture

```
                    ┌─────────────────────────────────────┐
                    │           INPUT: Meme Image          │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │   EasyOCR +     │  │  CLIP ViT-B/32  │  │  CLIP ViT-B/32  │
    │  Preprocessing  │  │  Image Encoder  │  │  Text Encoder   │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                    │
             ▼                    └─────────┬──────────┘
    ┌─────────────────┐                     │
    │  Extracted Text │           ┌──────────▼──────────┐
    └────────┬────────┘           │  Joint Embedding    │
             │                    │  Space (512-d each) │
      ┌──────┴──────┐             └──────────┬──────────┘
      │             │                        │
      ▼             ▼              ┌──────────▼──────────┐
┌──────────┐  ┌──────────┐        │  Fine-tuned          │
│ Twitter  │  │Dehate-   │        │  Classifier Head     │
│ RoBERTa  │  │ BERT     │        │  (1024→512→128→2)    │
│Sentiment │  │  Hate    │        └──────────┬──────────┘
└────┬─────┘  └────┬─────┘                   │
     │              │              ┌──────────▼──────────┐
     │              │              │  CLIP Hate Score    │
     │              └──────────────┤  + Zero-shot Cats   │
     │                             └──────────┬──────────┘
     │                                        │
     └──────────────────┬─────────────────────┘
                        │
              ┌──────────▼──────────┐
              │   Combined Output   │
              │  Sentiment | Hate   │
              │  Category | Sim     │
              └─────────────────────┘
```

---

## 🤖 Models Used

| Task | Model | Parameters | Why This Model |
|------|-------|-----------|----------------|
| OCR | EasyOCR | — | Handles meme fonts, curved text, low contrast |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment` | 125M | Trained on 124M tweets — matches meme language style |
| Hate Speech | `Hate-speech-CNERG/dehatebert-mono-english` | 110M | Purpose-built for hate detection, understands context |
| Categorization | `typeform/distilbert-base-uncased-mnli` | 66M | Zero-shot — no training needed, flexible labels |
| Multimodal | `openai/clip-vit-base-patch32` | 151M | Joint image-text embedding, understands meme context |
| Fine-tuned Hate | CLIP + Classifier Head (custom) | 21M trainable | Trained on Facebook Hateful Memes dataset |

---

##  Project Structure

```
memeIQ/
├── main.py                  # CLI entry point + analysis pipeline
├── config.py                # Central config — models, thresholds, paths
├── requirements.txt
│
├── models/
│   ├── ocr.py               # EasyOCR with contrast enhancement
│   ├── sentiment.py         # Twitter-RoBERTa sentiment classifier
│   ├── hate_speech.py       # Dehatebert with confidence thresholds
│   ├── categorizer.py       # DistilBERT zero-shot categorizer
│   ├── clip_model.py        # CLIP multimodal + fine-tuned classifier
│   └── clip_finetuned.pt    # Fine-tuned weights (Val F1: 0.82)
│
├── utils/
│   ├── image_utils.py       # Image loading, validation, OCR enhancement
│   ├── text_utils.py        # Text cleaning, truncation
│   ├── metrics.py           # F1, precision, recall, confusion matrix
│   └── logger.py            # Loguru-based logging
│
├── api/
│   ├── app.py               # FastAPI application
│   ├── routes.py            # /analyze, /hate-speech, /sentiment, /health
│   └── schemas.py           # Pydantic request/response models
│
├── ui/
│   └── app.py               # Streamlit dark-themed UI
│
├── data/
│   ├── pipeline.py          # HuggingFace dataset loading + DataLoaders
│   └── finetune.py          # CLIP fine-tuning training script
│
└── tests/
    ├── test_models.py        # Unit tests with mocks
    └── test_api.py           # FastAPI integration tests
```

---

##  Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Pranav080405/AI-based-Meme-Recognition-Classification-and-Hate-Speech-Detection.git
cd AI-based-Meme-Recognition-Classification-and-Hate-Speech-Detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run CLI analysis
```bash
# Fast mode (no CLIP)
python main.py --image your_meme.jpg --no-clip

# Full multimodal analysis
python main.py --image your_meme.jpg

# JSON output only
python main.py --image your_meme.jpg --json
```

### 4. Launch Streamlit UI
```bash
streamlit run ui/app.py
```

### 5. Start FastAPI server
```bash
uvicorn api.app:app --reload --port 8000
# Swagger docs → http://localhost:8000/docs
```

### 6. Run tests
```bash
pytest tests/ -v
```

---

## 📊 Results

### Fine-tuning on Facebook Hateful Memes Dataset

Fine-tuned CLIP ViT-B/32 + custom classifier head on 12,887 labeled memes.

| Epoch | Train Loss | Train F1 | Val Loss | Val F1 |
|-------|-----------|----------|----------|--------|
| 1 | 0.5593 | 0.7301 | 0.4333 | 0.7959 |
| 2 | 0.3549 | 0.8495 | 0.3786 | 0.8156 |
| 3 | 0.2398 | 0.9136 | 0.4077 | 0.8189 |
| **4** | **0.1578** | **0.9545** | **0.4365** | **0.8219 ✅** |
| 5 | 0.1129 | 0.9723 | 0.4633 | 0.8183 |

**Best checkpoint: Epoch 4 — Val F1 = 0.8219**

Training setup: Google Colab T4 GPU, batch size 32, AdamW optimizer, cosine LR scheduling, weighted cross-entropy loss (1:2 class weighting for imbalance), partial fine-tuning (only last 2 transformer layers + projection layers + classifier head — 14.3% of parameters trainable).

### Sample Output

```
══════════════════════════════════════════════════
    MemeIQ — Analysis Report
══════════════════════════════════════════════════

  Extracted Text
──────────────────────────────────────────────────
  "When father brings a new electric car for me
   First week: second week:"
  OCR confidence: 87%

  Sentiment
──────────────────────────────────────────────────
  Label: NEUTRAL  (84% confident)

  Hate Speech Detection
──────────────────────────────────────────────────
  Label: NON-HATEFUL
  Combined hate score: 23%
  CLIP hate score: 18%

  CLIP Multimodal Analysis
──────────────────────────────────────────────────
  Image class: a meme about dark humor (23%)
  Image↔Text similarity: 0.35

   Meme Category
──────────────────────────────────────────────────
  Category: dark humor (80% confident)

  Completed in 20.16s
══════════════════════════════════════════════════
```

---

##  API Reference

### `POST /analyze`
Full multimodal meme analysis.

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@meme.jpg" \
  -F "use_clip=true"
```

**Response:**
```json
{
  "ocr": { "text": "...", "confidence": 0.87, "empty": false },
  "sentiment": { "label": "neutral", "score": 0.84, "scores": {...} },
  "hate_speech": { "hateful": false, "hate_score": 0.23, "needs_review": false },
  "clip": { "image_text_similarity": 0.35, "image_classification": {...} },
  "category": { "category": "dark humor", "score": 0.80 },
  "elapsed_seconds": 20.16
}
```

### `POST /hate-speech`
Hate speech detection only (faster).

### `POST /sentiment`
Sentiment analysis only (faster).

### `GET /health`
Health check and model load status.

Full interactive docs at `http://localhost:8000/docs`

---

##  Key Technical Decisions

**Why CLIP over ResNet?**
ResNet classifies images into ImageNet categories (cat, dog, aircraft carrier) — useless for memes. CLIP understands open-ended text descriptions, so we can ask "is this a meme about gaming?" and get a meaningful answer.

**Why Twitter-RoBERTa over raw BERT?**
Raw BERT was pre-trained on Wikipedia and books — formal English. Meme text is informal, abbreviated, and sarcastic. Twitter-RoBERTa was fine-tuned on 124M tweets, which matches the linguistic style of memes much more closely.

**Why zero-shot categorization over a trained classifier?**
Training a category classifier requires thousands of labeled examples per category. Zero-shot MNLI lets us define categories in plain English and classify without any training data. We can add or change categories by editing a config file.

**Why partial fine-tuning over full fine-tuning?**
Full fine-tuning a 151M parameter model on 10k examples risks catastrophic forgetting and overfitting. Freezing the first 10 layers preserves general vision/language understanding while the last 2 layers + classifier head adapt to the hate detection task. This gave us Val F1 = 0.82 with only 14.3% of parameters trainable.

**Why ensemble Dehatebert + CLIP?**
Dehatebert is text-only — it has no idea what the image looks like. Our fine-tuned CLIP sees both modalities. Averaging their scores reduces false positives from either model alone while maintaining sensitivity to genuinely hateful content.

---

##  Roadmap

- [ ] GradCAM explainability — highlight image regions that triggered hate detection
- [ ] Attention visualization — show which words drove sentiment/hate scores
- [ ] Deploy on HuggingFace Spaces with Gradio
- [ ] Fine-tune on additional meme datasets (MemeCap, MultiOFF)
- [ ] LoRA fine-tuning for larger CLIP variants (ViT-L/14)
- [ ] Confidence-based human review queue integration
- [ ] Multi-language OCR and analysis support

---

##  Research

This project is the implementation behind a research paper on AI-based meme analysis authored by Pranav Madanu and Rohan Madanu, under the guidance of Dr. Mohan. The paper has been shortlisted for publication and explores the limitations of unimodal approaches to meme understanding and the advantages of multimodal fusion for hate speech detection in internet memes.

The research paper draft is available in this repository: [`Meme_classification_ResearchPaper_draft.pdf`](./Meme_classification_ResearchPaper_draft.pdf)

---

##  Tech Stack

- **ML Framework:** PyTorch 2.0
- **NLP:** HuggingFace Transformers 5.x
- **Vision:** CLIP (OpenAI), EasyOCR
- **API:** FastAPI + Uvicorn
- **UI:** Streamlit
- **Data:** HuggingFace Datasets
- **Logging:** Loguru
- **Testing:** Pytest
- **Training:** Google Colab T4 GPU

---

##  Authors

**Pranav Madanu** — ML pipeline, model integration, fine-tuning, API, UI

**Rohan Madanu** — Research paper, dataset analysis, evaluation

**Dr. Mohan** — Research guidance and supervision


