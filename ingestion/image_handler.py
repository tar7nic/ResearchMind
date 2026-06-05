import base64
from pathlib import Path


SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def load_image_as_base64(image_path: str) -> dict:
    """
    Reads an image from disk and returns a base64-encoded dict
    ready to be passed to GPT-4o vision.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {ext}. Supported: {SUPPORTED_FORMATS}")

    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return {
        "media_type": media_type_map[ext],
        "data": encoded,
    }


def encode_image_bytes(image_bytes: bytes, media_type: str = "image/png") -> dict:
    """
    Encodes raw image bytes (e.g. from an in-memory upload) as base64.
    Used when the image comes from Streamlit's file uploader directly.
    """
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "media_type": media_type,
        "data": encoded,
    }