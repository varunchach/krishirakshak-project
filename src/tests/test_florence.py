"""Tests for Florence-2 inference."""

import pytest
from unittest.mock import patch, MagicMock
from src.models.florence_inference import FlorenceInferenceService, DiagnosisResult


@patch("src.models.florence_inference.boto3")
def test_confidence_levels(mock_boto):
    with patch("builtins.open", MagicMock()):
        with patch("yaml.safe_load", return_value={
            "model": {"sagemaker_endpoint": "test"},
            "image": {"resize_to": 224},
        }):
            service = FlorenceInferenceService.__new__(FlorenceInferenceService)
            service.thresholds = {"high": 0.85, "medium": 0.60, "low": 0.40, "reject": 0.40}

            assert service._get_confidence_level(0.95) == "high"
            assert service._get_confidence_level(0.70) == "medium"
            assert service._get_confidence_level(0.45) == "low"
            assert service._get_confidence_level(0.30) == "reject"


def test_diagnosis_result_dataclass():
    result = DiagnosisResult(
        disease_name="Tomato Early Blight",
        confidence=0.92,
        confidence_level="high",
        treatment_english="Remove infected leaves.",
        raw_output="Tomato Early Blight. Treatment: ...",
        inference_time_ms=500.0,
    )
    assert result.disease_name == "Tomato Early Blight"
    assert result.confidence == 0.92
