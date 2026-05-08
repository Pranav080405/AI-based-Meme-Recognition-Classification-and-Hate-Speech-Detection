#  MemeIQ — AI-Powered Meme Recognition, Classification & Hate Speech Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade multimodal AI system for meme understanding, hate speech detection, sentiment analysis, and category classification.**

*Based on a shortlisted research paper authored by Pranav Madanu and Rohan Madanu, and Dr. B. Mohan Rao, KLH University.*

[Research Paper](#-research-paper) • [What Changed](#-from-paper-to-production--what-we-upgraded) • [Architecture](#-system-architecture) • [Models](#-models-used) • [Setup](#-quick-start) • [Results](#-results)

</div>

---

##  Research Paper

This repository is the direct implementation of the research paper:

> **"AI-based Meme Recognition, Classification, and Hate Speech Detection"**
> Pranav Madanu, Rohan Madanu, Badavath MohanRao
> KLH University — Shortlisted for Publication

The paper is available in this repository: [`Meme_classification_ResearchPaper_draft.pdf`](./Meme_classification_ResearchPaper_draft.pdf)

### What the Paper Proposed

The research identified a critical gap in content moderation: over 60% of hate speech on platforms like Twitter and Facebook is embedded in memes, yet existing tools analyze text and images in isolation — completely missing the contextual interplay between them.

The paper proposed a multimodal AI framework with three phases:

**1. Recognition** — Detecting memes in social media streams using EasyOCR for text extraction.

**2. Classification** — Categorizing memes by theme (politics, humour, motivation, offensive, neutral) using a dual-pathway deep learning model: ResNet-50 for visual features fused with BERT for text features via an attention-weighting mechanism into a 512-dimensional joint embedding.

**3. Harm Detection** — Identifying hate speech using keyword-based filtering alongside BERT sentiment classification.

### Paper Results

The research achieved strong classification performance on a curated dataset of 7,000 memes (Hateful Memes, MAMI, Dank Memes, Memotion), annotated by three independent annotators with Cohen's kappa κ=0.82:

| Task | Metric | Score |
|------|--------|-------|
| Thematic Classification | Overall Accuracy | 85.4% (±1.2%) |
| Humour Category | Precision / Recall | 0.87 / 0.82 |
| Politics Category | Precision / Recall | 0.91 / 0.88 |
| Sentiment Analysis | Accuracy | 81.2% |
| Sarcasm Detection | F1 | 0.77 |
| Multimodal vs Text-only | F1 improvement | +8.9% |

The ablation study was particularly significant — it confirmed the core thesis: multimodal fusion (F1=0.86) consistently outperformed text-only (F1=0.79) and image-only (F1=0.72) approaches, validating the case for joint visual-textual analysis.

---

## 🔄 From Paper to Production — What We Upgraded

The paper established the research foundation. This repository takes every component and upgrades it to production-grade quality, replacing prototype implementations with state-of-the-art models and a full software stack.

### 1. Sentiment Analysis: Raw BERT → Twitter-RoBERTa

| | Paper | This Repo |
|--|-------|-----------|
| Model | `bert-base-uncased` (not fine-tuned) | `cardiffnlp/twitter-roberta-base-sentiment` |
| Training data | Wikipedia + BooksCorpus | 124M tweets |
| Labels | positive, negative, neutral, sarcastic | positive, neutral, negative |
| Output | Arbitrary logits (no calibration) | Calibrated probabilities with confidence scores |

**Why:** Raw BERT was pre-trained on formal English. Meme text is informal, abbreviated, ironic, and sarcastic — linguistically much closer to Twitter than Wikipedia. Twitter-RoBERTa was specifically fine-tuned on 124M tweets, making it dramatically better suited for meme language.

---

### 2. Hate Speech Detection: Keyword List → Dehatebert + Fine-tuned CLIP Ensemble

| | Paper | This Repo |
|--|-------|-----------|
| Method | Keyword matching (`['hate', 'kill', 'attack'...]`) | `Hate-speech-CNERG/dehatebert-mono-english` + fine-tuned CLIP |
| Context awareness | None — purely lexical | Full sentence context + image context |
| Confidence | Binary yes/no | Continuous score 0–1 with human review flags |
| Multimodal | No | Yes — CLIP sees both image and text simultaneously |

**Why:** Keyword detection has catastrophic false positive rates. "Kill it on stage" and "destroy them in chess" would both be flagged. Dehatebert understands sentence context. More critically, we added a fine-tuned CLIP classifier that sees the image too — because whether something is hateful often depends entirely on visual context, not just the words.

---

### 3. Image Analysis: ResNet-50 (ImageNet) → CLIP ViT-B/32

| | Paper | This Repo |
|--|-------|-----------|
| Model | ResNet-50 pretrained on ImageNet | `openai/clip-vit-base-patch32` |
| Output | ImageNet category (e.g., "comic book", "carton") | Open-ended text descriptions (e.g., "a meme about dark humor") |
| Image-text interaction | Late fusion via attention weighting | Shared embedding space — true multimodal understanding |
| Flexibility | Fixed 1000 ImageNet classes | Any text-described category, no retraining needed |

**Why:** ResNet classifying memes into ImageNet categories produces meaningless results for this domain. CLIP was trained on 400M image-text pairs and understands open-ended language descriptions. When we ask "is this a meme about gaming?", CLIP gives a meaningful answer. ResNet cannot.

---

### 4. Meme Categorization: Random Assignment → Zero-Shot DistilBERT-MNLI

| | Paper | This Repo |
|--|-------|-----------|
| Method | `np.random.choice(meme_categories)` ← literally random | `typeform/distilbert-base-uncased-mnli` zero-shot NLI |
| Accuracy | 0% (random) | ~80% confidence on clear categories |
| Flexibility | Fixed hardcoded list | Edit `config.py` to add/change categories instantly |

**Why:** The paper's original implementation was a placeholder for future work. Zero-shot NLI classification uses natural language entailment to match meme text against category descriptions without any labeled training data.

---

### 5. Fine-Tuning on Hateful Memes Dataset (New — Not in Paper)

The paper proposed using the Hateful Memes dataset for evaluation but did not fine-tune on it. We implemented full fine-tuning:

- **Architecture:** CLIP ViT-B/32 backbone + custom 3-layer classifier head (1024→512→128→2)
- **Strategy:** Partial fine-tuning — frozen layers 1–9, trainable layers 10–11 + projection + classifier (14.3% of parameters)
- **Dataset:** Facebook Hateful Memes — 12,887 training memes, 3,000 test memes
- **Hardware:** Google Colab T4 GPU, batch size 32, 5 epochs
- **Best checkpoint:** Epoch 4, Val F1 = **0.8219**
- **Ensemble:** Fine-tuned CLIP score averaged with Dehatebert for final hate classification

---

### 6. Software Architecture: Single Notebook → Production System

The paper's implementation was a single Colab notebook (~150 lines). This repository is a fully modular production system:

| Component | Paper | This Repo |
|-----------|-------|-----------|
| Structure | Single `.py` file | `models/`, `utils/`, `api/`, `data/`, `tests/` |
| API | None | FastAPI with Pydantic schemas, Swagger docs |
| UI | None | Streamlit dark-themed interactive interface |
| Logging | `print()` statements | Loguru with rotation, file + console handlers |
| Error handling | None | Input validation, confidence thresholds, human review flags |
| Testing | None | Pytest unit + integration tests with mocks |
| Data pipeline | Manual | HuggingFace Datasets + PyTorch DataLoaders |

---

##  System Architecture

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
    │  Contrast Enh.  │  │  Image Encoder  │  │  Text Encoder   │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                    │
             ▼                    └─────────┬──────────┘
    ┌─────────────────┐                     │
    │  Extracted Text │          ┌──────────▼──────────┐
    └────────┬────────┘          │  Shared Embedding   │
             │                   │  Space (512-d each) │
      ┌──────┴──────┐            └──────────┬──────────┘
      │             │                       │
      ▼             ▼             ┌──────────▼──────────┐
┌──────────┐  ┌──────────┐       │  Fine-tuned          │
│ Twitter  │  │Dehate-   │       │  Classifier Head     │
│ RoBERTa  │  │ BERT     │       │  Val F1 = 0.8219     │
│Sentiment │  │  Hate    │       └──────────┬──────────┘
└────┬─────┘  └────┬─────┘                  │
     │              │             ┌──────────▼──────────┐
     │              └─────────────┤  CLIP Hate Score +  │
     │                            │  Zero-shot Category │
     │                            └──────────┬──────────┘
     └─────────────────┬─────────────────────┘
                       │
             ┌──────────▼──────────┐
             │   Combined Output   │
             │  Sentiment | Hate   │
             │  Category | Sim     │
             └─────────────────────┘
```

---

## 🤖 Models Used

| Task | Model | Parameters | Why |
|------|-------|-----------|-----|
| OCR | EasyOCR | — | Handles meme fonts, curved text, low contrast |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment` | 125M | Trained on 124M tweets — matches meme language |
| Hate Speech | `Hate-speech-CNERG/dehatebert-mono-english` | 110M | Purpose-built hate detection with sentence context |
| Categorization | `typeform/distilbert-base-uncased-mnli` | 66M | Zero-shot — no training data needed |
| Multimodal | `openai/clip-vit-base-patch32` | 151M | Joint image-text embedding space |
| Fine-tuned Hate | CLIP + Classifier Head (custom) | 21M trainable | Trained on Facebook Hateful Memes (Val F1=0.82) |

---

## 📁 Project Structure

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
│   └── clip_model.py        # CLIP multimodal + fine-tuned classifier
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

## ⚡ Quick Start

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
# Fast mode — no CLIP
python main.py --image your_meme.jpg --no-clip

# Full multimodal pipeline
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
# Swagger UI → http://localhost:8000/docs
```

### 6. Run tests
```bash
pytest tests/ -v
```

---

##  Fine-Tuning Results

CLIP ViT-B/32 + classifier head fine-tuned on Facebook Hateful Memes dataset. Google Colab T4 GPU, batch size 32, AdamW + cosine LR scheduling, weighted cross-entropy loss (1:2 for class imbalance), partial fine-tuning (14.3% of parameters trainable).

| Epoch | Train Loss | Train F1 | Val Loss | Val F1 |
|-------|-----------|----------|----------|--------|
| 1 | 0.5593 | 0.7301 | 0.4333 | 0.7959 |
| 2 | 0.3549 | 0.8495 | 0.3786 | 0.8156 |
| 3 | 0.2398 | 0.9136 | 0.4077 | 0.8189 |
| **4** | **0.1578** | **0.9545** | **0.4365** | **0.8219 ✅** |
| 5 | 0.1129 | 0.9723 | 0.4633 | 0.8183 |

**Best checkpoint: Epoch 4 — Val F1 = 0.8219**

---

##  API Reference

### `POST /analyze` — Full multimodal analysis
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@meme.jpg" -F "use_clip=true"
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

### `POST /hate-speech` — Hate detection only (faster)
### `POST /sentiment` — Sentiment only (faster)
### `GET /health` — Health check + model status

Full interactive docs: `http://localhost:8000/docs`

---

## 🔮 Roadmap

- [ ] GradCAM explainability — highlight image regions driving hate detection
- [ ] Attention visualization — show which words drove sentiment scores
- [ ] Deploy on HuggingFace Spaces with Gradio
- [ ] Dynamic content moderation with live Reddit/Twitter meme fetching (proposed in paper §6.8)
- [ ] Fine-tune on additional datasets (MemeCap, MultiOFF, MAMI)
- [ ] LoRA fine-tuning for CLIP ViT-L/14
- [ ] Multi-language OCR and analysis

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


