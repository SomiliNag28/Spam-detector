![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)

# SMS Spam Detection API — ML Inference Service
A modular machine learning inference service that classifies SMS messages
as spam or ham using Naive Bayes and TF-IDF, served via a REST API.

## Project Structure

spam_detector/
├── data/               # Raw dataset (not committed to Git)
├── models/             # Trained model artifacts
├── src/
│   ├── explore.py      # Dataset exploration
│   ├── preprocess.py   # Text cleaning and feature engineering
│   └── train.py        # Model training and evaluation
├── api/
│   └── main.py         # FastAPI inference service
├── tests/
│   └── test_api.py     # Automated test suite
├── requirements.txt
└── README.md

## Tech Stack

- Python 3.10
- scikit-learn (Naive Bayes, TF-IDF)
- FastAPI + Uvicorn
- Joblib
- Pytest

## Features

- SMS spam classification using Machine Learning
- TF-IDF vectorization for text feature extraction
- Multinomial Naive Bayes classifier
- REST API built with FastAPI
- Automated API tests with Pytest
- Interactive Swagger documentation
- Production-style project structure

## Example API Request

```json
{
  "message": "Congratulations! You won a FREE iPhone!"
}
```
## Example API Response

```json
{
  "prediction": "spam"
}
```

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 97.94% |
| Spam Precision | 97.0% |
| Spam Recall | 87.2% |
| F1 Score (macro) | 0.9535 |

Evaluated on 1115 held-out test messages with a 80/20 stratified split.

## Quickstart

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd spam_detector
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Add the dataset

Download `spam.csv` from Kaggle and place it in `data/spam.csv`.
Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

### 3. Train the model

```bash
python src/train.py
```

### 4. Start the API server

```bash
uvicorn api.main:app --reload
```

### 5. Test the API

Visit http://127.0.0.1:8000/docs for the interactive API documentation.

Or use curl:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Congratulations! You won a FREE prize!\"}"
```

## Running Tests

```bash
pytest tests/test_api.py -v
```