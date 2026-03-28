"""Tests for API endpoints."""

import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "KrishiRakshak"
    assert "version" in data


def test_health():
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "model_version" in data
    assert "uptime_seconds" in data


def test_languages():
    resp = client.get("/v1/languages")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["languages"]) > 0
    codes = [l["code"] for l in data["languages"]]
    assert "en-IN" in codes
    assert "hi-IN" in codes


def test_diagnose_no_file():
    resp = client.post("/v1/diagnose")
    assert resp.status_code == 422  # Validation error


def test_diagnose_empty_file():
    resp = client.post(
        "/v1/diagnose",
        files={"image": ("test.jpg", b"", "image/jpeg")},
        data={"language": "en-IN"},
    )
    assert resp.status_code == 400


def test_diagnose_wrong_format():
    resp = client.post(
        "/v1/diagnose",
        files={"image": ("test.txt", b"hello", "text/plain")},
        data={"language": "en-IN"},
    )
    assert resp.status_code == 400


@patch("src.api.routes.get_pipeline")
def test_diagnose_success(mock_pipeline, sample_image_bytes):
    mock_result = {
        "request_id": "test-123",
        "disease": {
            "name": "Tomato Early Blight",
            "crop": "Tomato",
            "severity": "moderate",
            "confidence": 0.92,
            "confidence_level": "high",
        },
        "treatment": {
            "english": "Remove infected leaves.",
            "translated": "संक्रमित पत्तियों को हटाएं।",
            "language": "hi-IN",
        },
        "audio": None,
        "metadata": {
            "request_id": "test-123",
            "model_version": "1.1.0",
            "inference_time_ms": 500,
            "total_time_ms": 1200,
            "timestamp": "2026-03-28T10:00:00Z",
        },
    }
    mock_pipeline.return_value.run.return_value = mock_result

    resp = client.post(
        "/v1/diagnose",
        files={"image": ("leaf.jpg", sample_image_bytes, "image/jpeg")},
        data={"language": "hi-IN"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["disease"]["name"] == "Tomato Early Blight"
    assert data["disease"]["confidence"] > 0.9


def test_feedback():
    with patch("src.api.routes.get_feedback") as mock_fb:
        mock_fb.return_value.submit_feedback = MagicMock()
        resp = client.post("/v1/feedback", json={
            "request_id": "test-123",
            "is_correct": True,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "recorded"
