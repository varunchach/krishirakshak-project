"""Image preprocessing utilities."""

import io
import logging

from PIL import Image, ExifTags

logger = logging.getLogger(__name__)

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_SIZE_BYTES = 5 * 1024 * 1024


def validate_image(image_bytes: bytes) -> tuple[bool, str]:
    """Validate image format and size."""
    if len(image_bytes) == 0:
        return False, "Empty file"
    if len(image_bytes) > MAX_SIZE_BYTES:
        return False, f"File too large ({len(image_bytes) / 1e6:.1f}MB). Max: 5MB"
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.format not in ALLOWED_FORMATS:
            return False, f"Unsupported format: {img.format}. Use JPEG or PNG."
        return True, "valid"
    except Exception as e:
        return False, f"Invalid image: {e}"


def preprocess(image_bytes: bytes, target_size: int = 224) -> bytes:
    """Resize, strip EXIF, normalize for model input."""
    img = Image.open(io.BytesIO(image_bytes))

    # Strip EXIF (privacy)
    data = list(img.getdata())
    clean = Image.new(img.mode, img.size)
    clean.putdata(data)

    # Resize
    clean = clean.convert("RGB").resize((target_size, target_size), Image.LANCZOS)

    buf = io.BytesIO()
    clean.save(buf, format="JPEG", quality=95)
    return buf.getvalue()
