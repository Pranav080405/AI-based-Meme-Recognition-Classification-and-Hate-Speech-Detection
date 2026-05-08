# 🧠 MemeIQ — AI-Powered Meme Analysis & Hate Speech Detection

A production-grade multimodal AI system for meme understanding, sentiment analysis, hate speech detection, and category classification.

---

## 🏗️ Architecture

```
meme_analyzer/
├── main.py                  # CLI entry point
├── config.py                # Central config & model names
├── requirements.txt
│
├── models/
│   ├── __init__.py
│   ├── ocr.py               # EasyOCR / PaddleOCR text extraction
│   ├── sentiment.py         # Twitter-RoBERTa sentiment model
│   ├── hate_speech.py       # Dehatebert hate speech classifier
│   ├── categorizer.py       # BART zero-shot meme categorizer
│   └── clip_model.py        # CLIP multimodal embeddings
│
├── utils/
│   ├── __init__.py
│   ├── image_utils.py       # Image preprocessing & enhancement
│   ├── text_utils.py        # Text cleaning helpers
│   ├── metrics.py           # Evaluation: F1, precision, recall, confusion matrix
│   └── logger.py            # Centralized logging
│
├── api/
│   ├── __init__.py
│   ├── app.py               # FastAPI application
│   ├── routes.py            # API endpoints
│   └── schemas.py           # Pydantic request/response models
│
├── ui/
│   └── app.py               # Streamlit UI
│
├── data/
│   ├── pipeline.py          # Dataset loading & preprocessing
│   └── splits.py            # Train/val/test splitting
│
└── tests/
    ├── test_models.py
    └── test_api.py
```

---

## 🚀 Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run CLI analysis
```bash
python main.py --image path/to/meme.jpg
```

### Start FastAPI server
```bash
uvicorn api.app:app --reload --port 8000
```

### Launch Streamlit UI
```bash
streamlit run ui/app.py
```

---

## 🤖 Models Used

| Task | Model | Why |
|------|-------|-----|
| OCR | EasyOCR | Fast, accurate on meme fonts |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment` | Twitter-trained, matches meme tone |
| Hate Speech | `Hate-speech-CNERG/dehatebert-mono-english` | Best open-source hate detector |
| Categorization | `facebook/bart-large-mnli` (zero-shot) | No training needed, flexible labels |
| Multimodal | `openai/clip-vit-base-patch32` | Unified image+text embeddings |

---

## 📊 Evaluation

Run metrics on a labeled dataset:
```bash
python -m utils.metrics --predictions preds.json --labels labels.json
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Full meme analysis |
| `POST` | `/hate-speech` | Hate speech only |
| `POST` | `/sentiment` | Sentiment only |
| `GET`  | `/health`  | Health check |

---

## 🔮 Roadmap

- [ ] Fine-tune on Facebook Hateful Memes dataset
- [ ] GradCAM explainability for image regions
- [ ] Attention visualization for text
- [ ] Confidence thresholds + human review queue
- [ ] HuggingFace Spaces deployment
- [ ] Ensemble scoring across all models
