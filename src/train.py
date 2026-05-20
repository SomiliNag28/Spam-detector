# Model training and evaluation for SMS spam detection.

import os
import sys
import joblib
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")

# Where trained artifacts will be saved
MODELS_DIR       = os.path.join(BASE_DIR, "models")
MODEL_PATH       = os.path.join(MODELS_DIR, "spam_model.joblib")
VECTORIZER_PATH  = os.path.join(MODELS_DIR, "vectorizer.joblib")

sys.path.insert(0, BASE_DIR)

from src.explore     import load_dataset
from src.preprocess  import preprocess_data

ALPHA = 0.1


def train_model(X_train, y_train) -> MultinomialNB:

    print("[Train] Fitting Multinomial Naive Bayes...")
    model = MultinomialNB(alpha=ALPHA)
    model.fit(X_train, y_train)
    print("  [OK]  Model fitted.")
    return model


def evaluate_model(model, X_test, y_test) -> None:
   
    print("\n" + "=" * 55)
    print("  EVALUATION REPORT")
    print("=" * 55)

    y_pred = model.predict(X_test)

    # ── Overall accuracy ──────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy  : {acc * 100:.2f}%")
    print(  "  (Note: accuracy alone is misleading on imbalanced data)")

    # ── F1 score (macro) ──────────────────────────────────────────────────
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n  F1 Score  : {f1:.4f}  (macro average)")

    # ── Per-class report ──────────────────────────────────────────────────
    print("\n  Per-Class Breakdown:")
    print("  " + "-" * 50)
    report = classification_report(
        y_test, y_pred,
        target_names=["ham (0)", "spam (1)"],
        digits=4
    )
    # Indent each line for clean formatting
    for line in report.splitlines():
        print("  " + line)

    # ── Confusion matrix ──────────────────────────────────────────────────
    print("\n  Confusion Matrix:")
    print("  " + "-" * 50)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print(f"\n                   Predicted HAM    Predicted SPAM")
    print(f"  Actual HAM    [       {tn:>5}      |       {fp:>5}       ]")
    print(f"  Actual SPAM   [       {fn:>5}      |       {tp:>5}       ]")

    print(f"\n  True  Negatives (ham correct)   : {tn}")
    print(f"  False Positives (ham as spam)   : {fp}  <- real emails wrongly blocked")
    print(f"  False Negatives (spam as ham)   : {fn}  <- spam that got through")
    print(f"  True  Positives (spam correct)  : {tp}")

    # ── Interpretation ────────────────────────────────────────────────────
    print("\n  Interpretation:")
    print("  " + "-" * 50)
    spam_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    spam_precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"  Of {tp + fn} actual spam messages:")
    print(f"    - Caught   : {tp} ({spam_recall*100:.1f}% recall)")
    print(f"    - Missed   : {fn} ({(1-spam_recall)*100:.1f}% slipped through)")
    print(f"  Spam precision : {spam_precision*100:.1f}% of spam flags were correct")

    print("\n" + "=" * 55)


def save_artifacts(model, vectorizer) -> None:

    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    model_size = os.path.getsize(MODEL_PATH)      / 1024
    vec_size   = os.path.getsize(VECTORIZER_PATH) / 1024

    print("\n[Save] Artifacts written to models/")
    print(f"  spam_model.joblib  : {model_size:.1f} KB")
    print(f"  vectorizer.joblib  : {vec_size:.1f} KB")
    print("  [OK]  Both artifacts saved successfully.")


def main():
   
    print("\n" + "=" * 55)
    print("  SMS SPAM DETECTOR — TRAINING PIPELINE")
    print("=" * 55 + "\n")

    # Step 1 — Load
    print("[Step 1/4] Loading dataset...")
    df = load_dataset(DATA_PATH)

    # Step 2 — Preprocess
    print("\n[Step 2/4] Preprocessing...")
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)

    # Step 3 — Train
    print("\n[Step 3/4] Training model...")
    model = train_model(X_train, y_train)

    # Step 4 — Evaluate
    print("\n[Step 4/4] Evaluating model...")
    evaluate_model(model, X_test, y_test)

    # Save
    save_artifacts(model, vectorizer)

    print("\n  TRAINING COMPLETE.")
    print("  Next step: build the FastAPI inference service.\n")


if __name__ == "__main__":
    main()