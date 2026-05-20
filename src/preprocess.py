# Text preprocessing and feature engineering for SMS spam detection.

import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42      # Seed for reproducibility — same split every run
TEST_SIZE     = 0.20    # 20% of data reserved for testing
MAX_FEATURES  = 5000    # Vocabulary size cap for TF-IDF
LABEL_MAP     = {"ham": 0, "spam": 1}  # Explicit mapping — no guessing


# ── Step 1: Text Cleaning ─────────────────────────────────────────────────────
def clean_text(text: str) -> str:

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"\d+", "", text)               # remove digits
    text = re.sub(r"[^\w\s]", "", text)           # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()      # collapse whitespace
    return text


# ── Step 2: Label Encoding ────────────────────────────────────────────────────
def encode_labels(labels: pd.Series) -> np.ndarray:
    
    unexpected = set(labels.unique()) - set(LABEL_MAP.keys())
    if unexpected:
        raise ValueError(
            f"Unexpected label values found: {unexpected}. "
            f"Expected only: {set(LABEL_MAP.keys())}"
        )

    return labels.map(LABEL_MAP).values


# ── Step 3: Train/Test Split ──────────────────────────────────────────────────
def split_data(
    X: pd.Series,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y            # <-- critical for imbalanced data
    )
    return X_train, X_test, y_train, y_test


# ── Step 4: TF-IDF Vectorization ─────────────────────────────────────────────
def build_vectorizer(max_features: int = MAX_FEATURES) -> TfidfVectorizer:
    
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )


# ── Main Pipeline Function ────────────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame):
   
    print("[Preprocess] Step 1/4 — Cleaning text...")
    df = df.dropna(subset=["label", "message"])   # drop any null rows
    df["cleaned"] = df["message"].apply(clean_text)

    print("[Preprocess] Step 2/4 — Encoding labels...")
    y = encode_labels(df["label"])

    print("[Preprocess] Step 3/4 — Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df["cleaned"], y)

    print("[Preprocess] Step 4/4 — Fitting TF-IDF vectorizer...")
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)  # fit + transform train
    X_test_tfidf  = vectorizer.transform(X_test)       # transform only test

    # ── Summary report ───────────────────────────────────────────────────────
    print("\n  [OK]  Preprocessing complete.")
    print(f"        Training samples : {X_train_tfidf.shape[0]}")
    print(f"        Test samples     : {X_test_tfidf.shape[0]}")
    print(f"        Vocabulary size  : {X_train_tfidf.shape[1]}")
    print(f"        Train spam count : {y_train.sum()}")
    print(f"        Test spam count  : {y_test.sum()}")

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer