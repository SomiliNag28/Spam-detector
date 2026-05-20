import sys
import os
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from api.main import app
from src.preprocess import clean_text


# ── Test Client ───────────────────────────────────────────────────────────────
client = TestClient(app)


# ══════════════════════════════════════════════════════
# SECTION 1 — Unit Tests: clean_text()
# ══════════════════════════════════════════════════════

class TestCleanText:

    def test_converts_to_lowercase(self):
        assert clean_text("FREE PRIZE") == "free prize"

    def test_removes_punctuation(self):
        result = clean_text("Hello!!! How are you???")
        assert "!" not in result
        assert "?" not in result

    def test_removes_numbers(self):
        result = clean_text("Call 08001234567 now")
        assert "0" not in result
        assert "8" not in result

    def test_removes_urls(self):
        result = clean_text("Visit http://win.com for prize")
        assert "http" not in result
        assert "win.com" not in result

    def test_collapses_whitespace(self):
        result = clean_text("hello   world")
        assert "  " not in result

    def test_handles_empty_string(self):
        result = clean_text("")
        assert isinstance(result, str)

    def test_handles_already_clean_text(self):
        result = clean_text("hello world")
        assert result == "hello world"

    def test_spam_like_message(self):
        raw     = "FREE PRIZE!! Call 08001234 or visit http://spam.com NOW!"
        cleaned = clean_text(raw)
        assert cleaned == cleaned.lower()
        assert "!" not in cleaned
        assert "http" not in cleaned


# ══════════════════════════════════════════════════════
# SECTION 2 — Integration Tests: GET endpoints
# ══════════════════════════════════════════════════════

class TestRootEndpoint:

    def test_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_returns_service_name(self):
        response = client.get("/")
        assert "SMS Spam Detector API" in response.json()["service"]

    def test_returns_docs_path(self):
        response = client.get("/")
        assert response.json()["docs"] == "/docs"


class TestHealthEndpoint:

    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_is_ok(self):
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_model_is_loaded(self):
        response = client.get("/health")
        assert response.json()["model_loaded"] is True

    def test_version_present(self):
        response = client.get("/health")
        assert "version" in response.json()


# ══════════════════════════════════════════════════════
# SECTION 3 — Integration Tests: POST /predict
# ══════════════════════════════════════════════════════

class TestPredictEndpoint:

    # ── Spam detection ────────────────────────────────
    def test_detects_spam_prize_message(self):
        response = client.post("/predict", json={
            "message": "Congratulations! You have won a FREE prize. Call 08001234 NOW!"
        })
        assert response.status_code == 200
        assert response.json()["label"] == "spam"
        assert response.json()["is_spam"] is True

    def test_detects_spam_urgent_message(self):
        response = client.post("/predict", json={
            "message": "URGENT: Your mobile number has won 2000 pounds. Call now to claim!"
        })
        assert response.status_code == 200
        assert response.json()["label"] == "spam"

    # ── Ham detection ─────────────────────────────────
    def test_detects_ham_casual_message(self):
        response = client.post("/predict", json={
            "message": "Hey, are you coming to the meeting at 3pm today?"
        })
        assert response.status_code == 200
        assert response.json()["label"] == "ham"
        assert response.json()["is_spam"] is False

    def test_detects_ham_friendly_message(self):
        response = client.post("/predict", json={
            "message": "I will call you later tonight, going for dinner now"
        })
        assert response.status_code == 200
        assert response.json()["label"] == "ham"

    # ── Response structure ────────────────────────────
    def test_response_contains_all_fields(self):
        response = client.post("/predict", json={
            "message": "Test message"
        })
        data = response.json()
        assert "message"    in data
        assert "cleaned"    in data
        assert "label"      in data
        assert "label_id"   in data
        assert "confidence" in data
        assert "is_spam"    in data

    def test_confidence_is_between_0_and_1(self):
        response = client.post("/predict", json={
            "message": "Free prize winner call now"
        })
        confidence = response.json()["confidence"]
        assert 0.0 <= confidence <= 1.0

    def test_label_id_matches_label(self):
        response = client.post("/predict", json={
            "message": "Free entry win cash prize call now"
        })
        data = response.json()
        if data["label"] == "spam":
            assert data["label_id"] == 1
        else:
            assert data["label_id"] == 0

    def test_original_message_preserved(self):
        original = "Hey are you free tonight?"
        response = client.post("/predict", json={"message": original})
        assert response.json()["message"] == original

    # ── Input validation ──────────────────────────────
    def test_rejects_empty_message(self):
        response = client.post("/predict", json={"message": ""})
        assert response.status_code == 422

    def test_rejects_missing_message_field(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_rejects_wrong_data_type(self):
        response = client.post("/predict", json={"message": 12345})
        assert response.status_code == 422

    def test_rejects_message_too_long(self):
        long_message = "a" * 501
        response = client.post("/predict", json={"message": long_message})
        assert response.status_code == 422