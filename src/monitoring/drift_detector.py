"""Basic drift detection on incoming images vs training baseline."""

import io
import json
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect distribution drift in incoming images."""

    def __init__(self, baseline_path: str = "configs/image_baseline.json"):
        try:
            with open(baseline_path) as f:
                self.baseline = json.load(f)
        except FileNotFoundError:
            logger.warning(f"No baseline at {baseline_path}. Drift detection disabled.")
            self.baseline = None

    def compute_stats(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(image)
        return {
            "mean_brightness": float(arr.mean()),
            "std_brightness": float(arr.std()),
            "width": image.width,
            "height": image.height,
            "size_kb": len(image_bytes) / 1024,
        }

    def check_drift(self, image_bytes: bytes, threshold_std: float = 3.0) -> dict:
        if not self.baseline:
            return {"drift_detected": False, "reason": "no_baseline"}

        stats = self.compute_stats(image_bytes)
        alerts = []

        for metric in ["mean_brightness", "size_kb"]:
            if metric in self.baseline:
                baseline_mean = self.baseline[metric]["mean"]
                baseline_std = self.baseline[metric]["std"]
                deviation = abs(stats[metric] - baseline_mean) / max(baseline_std, 1e-6)
                if deviation > threshold_std:
                    alerts.append(f"{metric}: {deviation:.1f} std from baseline")

        return {
            "drift_detected": len(alerts) > 0,
            "alerts": alerts,
            "stats": stats,
        }
