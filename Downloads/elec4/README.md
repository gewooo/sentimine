# SentimentIQ — Product Review Intelligence Dashboard

A professional ML-powered dashboard for sentiment and emotion analysis of product reviews.

## Features

- **4 ML Algorithms**: Logistic Regression, Naive Bayes, SVM, Random Forest
- **Dual Classification**: Sentiment (Positive/Negative) + Emotion (Happy, Sadness, Fear, Love, Anger)
- **Real-time Analysis**: Type any text and get instant predictions with confidence scores
- **Batch CSV Upload**: Analyze up to 20 reviews at once from a CSV file
- **Algorithm Comparison**: See how all 4 models predict the same text side-by-side
- **Full Metrics**: Accuracy, Precision, Recall, F1 per class per model
- **Dataset Dashboard**: Visualize emotion/sentiment distributions, category breakdowns

## Prerequisites

- Python 3.8+ with pip
- Node.js 18+ with npm

## Quick Start

### Terminal 1 — Backend
```bash
cd backend
pip install fastapi uvicorn scikit-learn pandas numpy python-multipart
uvicorn main:app --reload --port 8000
```

### Terminal 2 — Frontend
```bash
cd frontend
npm install
npm run dev
```

### Open in browser
```
http://localhost:5173
```

## Usage

1. Open the app and click **"Upload CSV"** in the top right
2. Upload `PRDECT-ID_Dataset_Translated.csv`
3. Wait ~15 seconds for all 4 models to train
4. Go to **Analyze** page → type a review → click Analyze
5. Go to **Model Metrics** page to see full evaluation results

## Dataset Format

The CSV must contain these columns:
- `Customer Review (English)` — the review text
- `Sentiment` — Positive / Negative
- `Emotion` — Happy / Sadness / Fear / Love / Anger

## Model Performance (PRDECT-ID Dataset)

| Algorithm | Sentiment Acc | Emotion Acc |
|---|---|---|
| SVM | **94.4%** | 60.4% |
| Logistic Regression | 93.4% | **62.8%** |
| Naive Bayes | 92.7% | 58.6% |
| Random Forest | 92.4% | 61.1% |

## Tech Stack

- **Backend**: Python, FastAPI, scikit-learn, TF-IDF Vectorizer
- **Frontend**: React 18, Vite, Recharts, Axios
- **Fonts**: Syne, JetBrains Mono, Inter
