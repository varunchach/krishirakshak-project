"""
SageMaker inference handler for EfficientNet-B3 crop disease classifier.

Expected model.tar.gz layout:
    inference.py
    best_model.pth

Input:  raw image bytes (image/jpeg or image/png), Content-Type header forwarded
Output: {"disease": str, "confidence": float (0-1), "low_conf": bool}
"""

import io
import json
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Class order MUST match src/models/classifier.py TOP_5_DISEASES
IDX_TO_CLASS = {
    0: "Tomato Early Blight",
    1: "Tomato Late Blight",
    2: "Potato Late Blight",
    3: "Tomato Leaf Mold",
    4: "Corn Common Rust",
}
NUM_CLASSES = len(IDX_TO_CLASS)
LOW_CONF_THRESHOLD = 0.40

PREPROCESS = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def model_fn(model_dir: str):
    device = torch.device("cpu")

    net = models.efficientnet_b3(weights=None)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )

    weights_path = os.path.join(model_dir, "best_model.pth")
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.to(device)
    net.eval()
    return net


def input_fn(request_body, content_type: str):
    if content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise ValueError(f"Unsupported content type: {content_type}")
    return Image.open(io.BytesIO(request_body)).convert("RGB")


def predict_fn(image: Image.Image, model):
    tensor = PREPROCESS(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    confidence, idx = probs.max(0)
    confidence = confidence.item()
    return {
        "disease": IDX_TO_CLASS[idx.item()],
        "confidence": round(confidence, 4),
        "low_conf": confidence < LOW_CONF_THRESHOLD,
    }


def output_fn(prediction, accept: str):
    return json.dumps(prediction), "application/json"
