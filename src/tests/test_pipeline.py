"""Tests for image preprocessor and services."""

import pytest
from src.pipeline.image_preprocessor import validate_image, preprocess


def test_validate_valid_image(sample_image_bytes):
    valid, msg = validate_image(sample_image_bytes)
    assert valid is True
    assert msg == "valid"


def test_validate_empty():
    valid, msg = validate_image(b"")
    assert valid is False
    assert "Empty" in msg


def test_validate_invalid(invalid_file_bytes):
    valid, msg = validate_image(invalid_file_bytes)
    assert valid is False


def test_preprocess(sample_image_bytes):
    result = preprocess(sample_image_bytes, target_size=224)
    assert len(result) > 0
    assert len(result) < len(sample_image_bytes) * 2  # Shouldn't explode in size

    from PIL import Image
    import io
    img = Image.open(io.BytesIO(result))
    assert img.size == (224, 224)
    assert img.mode == "RGB"
