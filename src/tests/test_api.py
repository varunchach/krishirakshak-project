"""Tests for FastAPI endpoints."""

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
    assert data["status"] in {"healthy", "degraded"}
    assert "classifier" in data
    assert "embeddings" in data
    assert "faiss_index_size" in data


def test_diagnose_no_file():
    resp = client.post("/v1/diagnose")
    assert resp.status_code == 422


def test_diagnose_empty_file():
    resp = client.post(
        "/v1/diagnose",
        files={"image": ("test.jpg", b"", "image/jpeg")},
        data={"session_id": "test"},
    )
    assert resp.status_code == 400


def test_diagnose_wrong_format():
    resp = client.post(
        "/v1/diagnose",
        files={"image": ("test.txt", b"hello", "text/plain")},
        data={"session_id": "test"},
    )
    assert resp.status_code == 400


@patch("src.api.routes._get_classifier")
@patch("src.api.routes._get_agent")
@patch("src.api.routes.text_to_speech", return_value=b"fake-mp3")
@patch("src.api.routes._s3")
def test_diagnose_low_conf(mock_s3, mock_tts, mock_agent, mock_clf, sample_image_bytes):
    mock_clf.return_value = MagicMock()
    with patch("src.api.routes.predict", return_value={
        "disease": "Tomato Early Blight", "confidence": 25.0, "low_conf": True
    }):
        mock_s3.upload_audio.return_value = "https://s3.example.com/audio.mp3"
        resp = client.post(
            "/v1/diagnose",
            files={"image": ("leaf.jpg", sample_image_bytes, "image/jpeg")},
            data={"session_id": "s1"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "Confidence is too low" in data["answer"] or "low" in data["answer"].lower()
    assert data["request_id"]


@patch("src.api.routes._get_classifier")
@patch("src.api.routes._get_agent")
@patch("src.api.routes.agent_run", return_value="Apply Mancozeb spray every 7 days.")
@patch("src.api.routes.text_to_speech", return_value=b"fake-mp3")
@patch("src.api.routes._s3")
def test_diagnose_success(mock_s3, mock_tts, mock_run, mock_agent, mock_clf, sample_image_bytes):
    mock_clf.return_value = MagicMock()
    with patch("src.api.routes.predict", return_value={
        "disease": "Tomato Early Blight", "confidence": 91.5, "low_conf": False
    }):
        mock_s3.upload_audio.return_value = "https://s3.example.com/audio.mp3"
        resp = client.post(
            "/v1/diagnose",
            files={"image": ("leaf.jpg", sample_image_bytes, "image/jpeg")},
            data={"session_id": "s1"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "Tomato Early Blight" in data["answer"]
    assert data["session_id"] == "s1"
    assert data["request_id"]


@patch("src.api.routes._get_agent")
@patch("src.api.routes.agent_run", return_value="Apply Mancozeb spray.")
@patch("src.api.routes.text_to_speech", return_value=b"fake-mp3")
@patch("src.api.routes._s3")
def test_query_endpoint(mock_s3, mock_tts, mock_run, mock_agent):
    mock_s3.upload_audio.return_value = "https://s3.example.com/audio.mp3"
    resp = client.post("/v1/query", json={
        "query": "How to treat tomato blight?",
        "session_id": "s2",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["session_id"] == "s2"
    assert data["request_id"]


def test_feedback_endpoint():
    with patch("src.api.routes._feedback") as mock_fb:
        mock_fb.submit_feedback = MagicMock()
        resp = client.post("/v1/feedback", json={
            "request_id": "req-123",
            "is_correct": True,
            "comment": "Correct diagnosis",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "recorded"
    assert data["request_id"] == "req-123"
