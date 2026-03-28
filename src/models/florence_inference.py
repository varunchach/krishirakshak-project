"""
Florence-2 inference service — wraps SageMaker endpoint or local model.

Handles: image preprocessing, endpoint invocation, response parsing.
"""

import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass

import boto3
import yaml
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DiagnosisResult:
    disease_name: str
    confidence: float
    confidence_level: str  # high, medium, low
    treatment_english: str
    raw_output: str
    inference_time_ms: float


class FlorenceInferenceService:
    """Invoke Florence-2 on SageMaker for crop disease diagnosis."""

    def __init__(self, config_path: str = "configs/app_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        with open("configs/model_config.yaml") as f:
            self.model_config = yaml.safe_load(f)

        self.endpoint_name = self.config["model"]["sagemaker_endpoint"]
        self.thresholds = self.model_config["confidence_thresholds"]
        self.client = boto3.client("sagemaker-runtime")

        logger.info(f"Florence inference service initialized (endpoint: {self.endpoint_name})")

    def _preprocess_image(self, image_bytes: bytes) -> bytes:
        """Resize and validate image for Florence-2 input."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        target_size = self.config["image"]["resize_to"]
        image = image.resize((target_size, target_size), Image.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()

    def _parse_confidence(self, raw_output: str) -> float:
        """Extract confidence score from model output."""
        # Florence-2 fine-tuned output format:
        # "Tomato Early Blight. Severity: moderate. Symptoms: ..."
        # Confidence is inferred from model's logits (returned by SageMaker)
        # For now, use a heuristic based on output structure
        confidence_match = re.search(r"Confidence:\s*([\d.]+)", raw_output)
        if confidence_match:
            return float(confidence_match.group(1))
        # If model didn't output explicit confidence, default to medium
        return 0.75

    def _get_confidence_level(self, confidence: float) -> str:
        if confidence >= self.thresholds["high"]:
            return "high"
        elif confidence >= self.thresholds["medium"]:
            return "medium"
        elif confidence >= self.thresholds["low"]:
            return "low"
        return "reject"

    def _parse_output(self, raw_output: str, inference_time_ms: float) -> DiagnosisResult:
        """Parse Florence-2 generated text into structured result."""
        # Expected format: "Disease Name. Severity: X. Symptoms: Y. Immediate Action: Z..."
        parts = raw_output.split(". ", 1)
        disease_name = parts[0].strip() if parts else "Unknown"

        treatment = parts[1] if len(parts) > 1 else ""
        confidence = self._parse_confidence(raw_output)
        confidence_level = self._get_confidence_level(confidence)

        return DiagnosisResult(
            disease_name=disease_name,
            confidence=confidence,
            confidence_level=confidence_level,
            treatment_english=treatment,
            raw_output=raw_output,
            inference_time_ms=inference_time_ms,
        )

    def diagnose(self, image_bytes: bytes) -> DiagnosisResult:
        """Run diagnosis on a leaf image via SageMaker endpoint."""
        start_time = time.monotonic()

        try:
            processed_image = self._preprocess_image(image_bytes)

            payload = {
                "image": base64.b64encode(processed_image).decode("utf-8"),
                "prompt": "<CROP_DISEASE>",
            }

            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )

            result = json.loads(response["Body"].read().decode("utf-8"))
            raw_output = result.get("generated_text", "")
            confidence = result.get("confidence", 0.75)

            inference_time_ms = (time.monotonic() - start_time) * 1000

            diagnosis = self._parse_output(raw_output, inference_time_ms)
            # Override confidence if endpoint returned it
            if "confidence" in result:
                diagnosis.confidence = confidence
                diagnosis.confidence_level = self._get_confidence_level(confidence)

            logger.info(
                f"Diagnosis: {diagnosis.disease_name} "
                f"(confidence={diagnosis.confidence:.2f}, "
                f"latency={diagnosis.inference_time_ms:.0f}ms)"
            )
            return diagnosis

        except Exception as e:
            inference_time_ms = (time.monotonic() - start_time) * 1000
            logger.error(f"SageMaker inference failed: {e}", exc_info=True)
            raise RuntimeError(f"Model inference failed: {e}") from e


class LocalFlorenceInference:
    """Local inference for development and testing (no SageMaker)."""

    def __init__(self, model_path: str = "outputs/florence2_lora/best"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading local model from {model_path} on {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base-ft",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.to(self.device)
        self.model.eval()

        with open("configs/model_config.yaml") as f:
            self.model_config = yaml.safe_load(f)

        logger.info("Local model loaded successfully")

    def diagnose(self, image_bytes: bytes) -> DiagnosisResult:
        import torch

        start_time = time.monotonic()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(
            text="<CROP_DISEASE>",
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=3,
            )

        raw_output = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        inference_time_ms = (time.monotonic() - start_time) * 1000

        parts = raw_output.split(". ", 1)
        disease_name = parts[0].strip() if parts else "Unknown"
        treatment = parts[1] if len(parts) > 1 else ""

        return DiagnosisResult(
            disease_name=disease_name,
            confidence=0.85,  # Local inference doesn't return calibrated confidence
            confidence_level="medium",
            treatment_english=treatment,
            raw_output=raw_output,
            inference_time_ms=inference_time_ms,
        )
