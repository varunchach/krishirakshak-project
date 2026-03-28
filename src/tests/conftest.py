"""Shared test fixtures."""

import io
import pytest
from PIL import Image


@pytest.fixture
def sample_image_bytes():
    """Generate a small test image."""
    img = Image.new("RGB", (224, 224), (0, 128, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def large_image_bytes():
    """Generate an oversized image (>5MB)."""
    img = Image.new("RGB", (5000, 5000), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def invalid_file_bytes():
    """Non-image bytes."""
    return b"This is not an image file"
