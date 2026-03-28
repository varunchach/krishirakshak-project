"""Model configuration and disease class mappings."""

import json
from pathlib import Path

import yaml


def load_model_config(path: str = "configs/model_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_treatment_kb(path: str = "training/treatment_kb.json") -> dict:
    with open(path) as f:
        return json.load(f)["diseases"]


def get_disease_info(class_id: int, config: dict | None = None) -> dict:
    """Get disease info by class ID."""
    if config is None:
        config = load_model_config()
    return config["disease_classes"].get(class_id, {"name": "Unknown", "crop": "Unknown", "severity": "unknown"})


def get_treatment(disease_name: str, kb: dict | None = None) -> dict:
    """Get treatment info from knowledge base."""
    if kb is None:
        kb = load_treatment_kb()
    return kb.get(disease_name, {})
