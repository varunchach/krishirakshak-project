"""
EfficientNet-B3 crop disease classifier integration.

Production:
    FastAPI -> SageMaker endpoint -> prediction

Local/test:
    FastAPI/tests -> local PyTorch weights -> prediction
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from PIL import Image

logger = logging.getLogger(__name__)

TOP_5_DISEASES = [
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Potato Late Blight",
    "Tomato Leaf Mold",
    "Corn Common Rust",
]
IDX_TO_CLASS = {i: disease for i, disease in enumerate(TOP_5_DISEASES)}
NUM_CLASSES = len(TOP_5_DISEASES)
LOW_CONF_THRESHOLD = 0.40

DEFAULT_MODEL_PATH = "models_pkl/best_model.pth"
DEFAULT_BACKEND = "local"
DEFAULT_ENDPOINT_REGION = os.getenv("CLASSIFIER_SAGEMAKER_REGION", os.getenv("SAGEMAKER_REGION", "ap-south-1"))
DEFAULT_ENDPOINT_NAME = os.getenv("CLASSIFIER_SAGEMAKER_ENDPOINT", "krishirakshak-efficientnet-b3")


def _get_local_ml_modules():
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms

    return torch, nn, models, transforms


def _build_preprocess():
    _, _, _, transforms = _get_local_ml_modules()
    return transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _build_model():
    _, nn, models, _ = _get_local_ml_modules()
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


class SageMakerClassifierEndpoint:
    """Thin client for the hosted EfficientNet endpoint."""

    def __init__(self, endpoint_name: str, region_name: str):
        self.endpoint_name = endpoint_name
        self.region_name = region_name
        self.runtime = boto3.client(
            "sagemaker-runtime",
            region_name=region_name,
            config=BotoConfig(read_timeout=60, connect_timeout=10),
        )

    def predict(self, image_bytes: bytes, content_type: str = "image/jpeg") -> dict[str, Any]:
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType=content_type,
            Accept="application/json",
            Body=image_bytes,
        )
        payload = json.loads(response["Body"].read())
        return _normalize_prediction(payload)


def _normalize_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    disease = payload["disease"]
    confidence = float(payload["confidence"])
    if confidence <= 1.0:
        confidence *= 100

    confidence = round(confidence, 1)
    low_conf = payload.get("low_conf")
    if low_conf is None:
        low_conf = (confidence / 100) < LOW_CONF_THRESHOLD

    return {
        "disease": disease,
        "confidence": confidence,
        "low_conf": bool(low_conf),
    }


def _predict_local(model, image_bytes: bytes | None = None, image_path: str | None = None) -> dict[str, Any]:
    torch, _, _, _ = _get_local_ml_modules()
    preprocess = _build_preprocess()
    device = next(model.parameters()).device

    if image_bytes is not None:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    elif image_path:
        image = Image.open(image_path).convert("RGB")
    else:
        raise ValueError("Either image_bytes or image_path must be provided.")

    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        confidence, index = probabilities.max(0)

    disease = IDX_TO_CLASS[index.item()]
    confidence_pct = round(confidence.item() * 100, 1)
    logger.info(f"Predicted locally: {disease} ({confidence_pct}%)")
    return {
        "disease": disease,
        "confidence": confidence_pct,
        "low_conf": confidence.item() < LOW_CONF_THRESHOLD,
    }


def load_model(model_path: str | None = None, backend: str | None = None):
    """Load the configured classifier backend."""
    chosen_backend = (backend or os.getenv("CLASSIFIER_BACKEND", DEFAULT_BACKEND)).lower()
    if chosen_backend == "sagemaker":
        endpoint_name = os.getenv("CLASSIFIER_SAGEMAKER_ENDPOINT", DEFAULT_ENDPOINT_NAME)
        endpoint_region = os.getenv("CLASSIFIER_SAGEMAKER_REGION", DEFAULT_ENDPOINT_REGION)
        logger.info(f"Classifier configured for SageMaker endpoint {endpoint_name} ({endpoint_region})")
        return SageMakerClassifierEndpoint(endpoint_name=endpoint_name, region_name=endpoint_region)

    torch, _, _, _ = _get_local_ml_modules()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_path or os.getenv("CLASSIFIER_MODEL_PATH", DEFAULT_MODEL_PATH)

    if model_path.startswith("s3://"):
        bucket, key = model_path[5:].split("/", 1)
        local_path = "/tmp/best_model.pth"
        logger.info(f"Downloading model from s3://{bucket}/{key}")
        boto3.client("s3").download_file(bucket, key, local_path)
        model_path = local_path

    model = _build_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info(f"Classifier loaded on {device} from {model_path}")
    return model


def predict(
    model,
    image_path: str | None = None,
    image_bytes: bytes | None = None,
    content_type: str = "image/jpeg",
) -> dict[str, Any]:
    """Run inference against the configured backend."""
    if isinstance(model, SageMakerClassifierEndpoint):
        if image_bytes is None:
            if not image_path:
                raise ValueError("Either image_bytes or image_path must be provided.")
            image_bytes = Path(image_path).read_bytes()
        result = model.predict(image_bytes=image_bytes, content_type=content_type)
        logger.info(f"Predicted via SageMaker: {result['disease']} ({result['confidence']}%)")
        return result

    return _predict_local(model, image_bytes=image_bytes, image_path=image_path)


def get_backend_status(model=None) -> dict[str, Any]:
    """Return a lightweight health snapshot for the classifier backend."""
    backend = os.getenv("CLASSIFIER_BACKEND", DEFAULT_BACKEND).lower()
    if backend == "sagemaker":
        endpoint_name = os.getenv("CLASSIFIER_SAGEMAKER_ENDPOINT", DEFAULT_ENDPOINT_NAME)
        endpoint_region = os.getenv("CLASSIFIER_SAGEMAKER_REGION", DEFAULT_ENDPOINT_REGION)
        client = boto3.client("sagemaker", region_name=endpoint_region)
        try:
            description = client.describe_endpoint(EndpointName=endpoint_name)
            status = description["EndpointStatus"]
        except Exception as exc:
            status = f"unreachable: {exc.__class__.__name__}"

        return {
            "backend": backend,
            "endpoint": endpoint_name,
            "region": endpoint_region,
            "status": status,
            "ready": status == "InService",
        }

    return {
        "backend": "local",
        "endpoint": None,
        "region": None,
        "status": "loaded" if model is not None else "not_loaded",
        "ready": model is not None,
    }
