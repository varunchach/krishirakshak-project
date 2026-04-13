import os
import json
import numpy as np
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model = None
tokenizer = None


def model_fn(model_dir):
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = ORTModelForFeatureExtraction.from_pretrained(
        model_dir,
        file_name="onnx/model_quantized.onnx",
    )
    return model


def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(request_body if isinstance(request_body, str) else request_body.decode("utf-8"))
    texts = payload["inputs"]
    if isinstance(texts, str):
        texts = [texts]
    return {"inputs": texts}


def predict_fn(data, model):
    texts = data["inputs"]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )
    outputs = model(**encoded)
    token_embeddings = outputs.last_hidden_state  # (batch, seq, dim)
    attention_mask = encoded["attention_mask"]   # (batch, seq)
    mask = attention_mask[:, :, np.newaxis].astype(float)
    pooled = (token_embeddings * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-12, None)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    embeddings = (pooled / norms).tolist()
    return {"embeddings": embeddings}


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
