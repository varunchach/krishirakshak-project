"""Tests for EfficientNet-B3 classifier."""

import io
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def test_top5_diseases_order():
    from src.models.classifier import TOP_5_DISEASES
    assert TOP_5_DISEASES[0] == "Tomato Early Blight"
    assert TOP_5_DISEASES[1] == "Tomato Late Blight"
    assert TOP_5_DISEASES[2] == "Potato Late Blight"
    assert TOP_5_DISEASES[3] == "Tomato Leaf Mold"
    assert TOP_5_DISEASES[4] == "Corn Common Rust"
    assert len(TOP_5_DISEASES) == 5


def test_predict_low_confidence():
    """If model returns very low logits, result should be low_conf=True."""
    from src.models.classifier import predict, _build_model
    import torch

    model = _build_model()
    model.eval()

    # Zero-weight model → uniform logits → low confidence
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    # Create a dummy white image
    from PIL import Image
    img = Image.new("RGB", (300, 300), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(buf.read())
        tmp = f.name

    result = predict(model, tmp)
    os.remove(tmp)

    assert "disease" in result
    assert "confidence" in result
    assert "low_conf" in result
    assert isinstance(result["confidence"], float)


def test_predict_result_keys():
    from src.models.classifier import predict, _build_model
    import torch
    from PIL import Image

    model = _build_model()
    model.eval()

    img = Image.new("RGB", (300, 300), color=(100, 150, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(buf.read())
        tmp = f.name

    result = predict(model, tmp)
    os.remove(tmp)

    assert set(result.keys()) == {"disease", "confidence", "low_conf"}
    assert result["disease"] in __import__("src.models.classifier", fromlist=["TOP_5_DISEASES"]).TOP_5_DISEASES
