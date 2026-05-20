# FastAPI inference service for SMS spam detection.

import os
import sys
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH      = os.path.join(BASE_DIR, "models", "spam_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.joblib")

sys.path.insert(0, BASE_DIR)

from src.preprocess import clean_text   # reuse exact same cleaning logic


# ── App Initialization ────────────────────────────────────────────────────────
app = FastAPI(
    title="SMS Spam Detector API",
    description=(
        "A machine learning inference service that classifies "
        "SMS messages as spam or ham (not spam). "
        "Built with scikit-learn, FastAPI, and Joblib."
    ),
    version="1.0.0",
)


# ── Model Loading ─────────────────────────────────────────────────────────────
def load_artifacts():
    
    for path in [MODEL_PATH, VECTORIZER_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model artifact not found: {path}\n"
                f"Please run 'python src/train.py' first."
            )

    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_artifacts()

# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Schema for incoming prediction requests."""
    message: str = Field(
        ...,                          # ... means required (no default)
        min_length=1,
        max_length=500,
        description="The SMS message text to classify.",
        examples=["Congratulations! You've won a free prize. Call now!"]
    )


class PredictResponse(BaseModel):
    """Schema for prediction responses."""
    message: str       = Field(description="The original input message.")
    cleaned: str       = Field(description="Text after preprocessing.")
    label: str         = Field(description="Predicted class: 'spam' or 'ham'.")
    label_id: int      = Field(description="Numeric label: 1=spam, 0=ham.")
    confidence: float  = Field(description="Model confidence score (0.0 to 1.0).")
    is_spam: bool      = Field(description="True if message is classified as spam.")


class HealthResponse(BaseModel):
    """Schema for health check responses."""
    status: str
    model_loaded: bool
    version: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", summary="Service info")
def root():

    return {
        "service"     : "SMS Spam Detector API",
        "version"     : "1.0.0",
        "docs"        : "/docs",
        "health"      : "/health",
        "predict"     : "/predict",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check"
)
def health():
  
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        version="1.0.0",
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify an SMS message"
)
def predict(request: PredictRequest):
  
    try:
        # Step 1 — Clean text
        cleaned = clean_text(request.message)

        # Step 2 — Transform to TF-IDF vector
        features = vectorizer.transform([cleaned])

        # Step 3 — Predict
        label_id    = int(model.predict(features)[0])
        label       = "spam" if label_id == 1 else "ham"

        # Step 4 — Get confidence score
        probabilities = model.predict_proba(features)[0]
        confidence    = float(probabilities[label_id])

        return PredictResponse(
            message    = request.message,
            cleaned    = cleaned,
            label      = label,
            label_id   = label_id,
            confidence = round(confidence, 4),
            is_spam    = label_id == 1,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )