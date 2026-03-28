"""
Diagnosis Pipeline — Orchestrates the full flow:
  Image → Florence-2 → (optional Bedrock) → Sarvam Translate → Sarvam TTS → Response
"""

import logging
import time
import uuid

import yaml

from src.models.florence_inference import DiagnosisResult, FlorenceInferenceService
from src.services.feedback_service import FeedbackService
from src.services.s3_service import S3Service
from src.services.sarvam_translate import SarvamTranslateService
from src.services.sarvam_tts import SarvamTTSService

logger = logging.getLogger(__name__)


class DiagnosisPipeline:
    """End-to-end crop disease diagnosis pipeline."""

    def __init__(self, config_path: str = "configs/app_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.florence = FlorenceInferenceService(config_path)
        self.translator = SarvamTranslateService()
        self.tts = SarvamTTSService()
        self.s3 = S3Service(config_path)
        self.feedback = FeedbackService(config_path)

        self.confidence_threshold = self.config["model"]["confidence_threshold"]
        self.use_bedrock_fallback = self.config["model"]["bedrock_fallback"]

        logger.info("Diagnosis pipeline initialized")

    def _augment_with_bedrock(self, diagnosis: DiagnosisResult) -> str:
        """Use Bedrock to generate richer treatment advice when Florence-2 is insufficient."""
        try:
            import json
            import boto3

            bedrock = boto3.client(
                "bedrock-runtime",
                region_name=self.config["model"]["bedrock_region"],
            )

            prompt = (
                f"You are an agricultural expert advising Indian farmers. "
                f"A farmer's crop has been diagnosed with: {diagnosis.disease_name}. "
                f"Provide detailed treatment advice in simple language. Include: "
                f"1) Immediate action needed "
                f"2) Recommended pesticide (available in India, with brand names) "
                f"3) Dosage and application method "
                f"4) Prevention tips for next season"
            )

            response = bedrock.invoke_model(
                modelId=self.config["model"]["bedrock_model_id"],
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.config["model"]["bedrock_max_tokens"],
                    "messages": [{"role": "user", "content": prompt}],
                }),
            )

            result = json.loads(response["body"].read())
            advice = result["content"][0]["text"]
            logger.info(f"Bedrock augmentation successful ({len(advice)} chars)")
            return advice

        except Exception as e:
            logger.warning(f"Bedrock fallback failed: {e}. Using Florence-2 output only.")
            return diagnosis.treatment_english

    def run(self, image_bytes: bytes, language: str = "en-IN",
            include_audio: bool = True) -> dict:
        """
        Execute full diagnosis pipeline.

        Returns structured response dict ready for API serialization.
        """
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        logger.info(f"Pipeline started: request_id={request_id}, language={language}")

        # Step 1: Upload image to S3 (async-safe, non-blocking on failure)
        image_key = ""
        try:
            image_key = self.s3.upload_image(image_bytes, request_id)
        except Exception as e:
            logger.warning(f"S3 upload failed (non-critical): {e}")

        # Step 2: Run Florence-2 inference
        try:
            diagnosis = self.florence.diagnose(image_bytes)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return self._error_response(request_id, "MODEL_UNAVAILABLE", str(e))

        # Step 3: Check confidence threshold
        if diagnosis.confidence < self.confidence_threshold:
            logger.warning(f"Low confidence: {diagnosis.confidence:.2f}")
            with open("configs/model_config.yaml") as f:
                model_config = yaml.safe_load(f)
            return {
                "request_id": request_id,
                "disease": {
                    "name": "uncertain",
                    "confidence": diagnosis.confidence,
                    "confidence_level": "low",
                },
                "treatment": {
                    "english": model_config["responses"]["uncertain"],
                },
                "audio": None,
                "metadata": self._build_metadata(request_id, start_time, diagnosis),
            }

        # Step 4: Augment with Bedrock if needed
        treatment_en = diagnosis.treatment_english
        if self.use_bedrock_fallback and self._should_use_bedrock(diagnosis):
            treatment_en = self._augment_with_bedrock(diagnosis)

        # Step 5: Translate (graceful degradation)
        translation = {"translated_text": treatment_en, "language": "en-IN", "latency_ms": 0}
        if language != "en-IN":
            try:
                translation = self.translator.translate(treatment_en, language)
            except Exception as e:
                logger.warning(f"Translation failed, returning English: {e}")

        # Step 6: TTS (graceful degradation)
        audio_result = None
        if include_audio:
            try:
                tts_text = translation["translated_text"]
                audio_result = self.tts.synthesize(tts_text, language)
                if audio_result.get("error"):
                    audio_result = None
            except Exception as e:
                logger.warning(f"TTS failed, skipping audio: {e}")

        # Step 7: Log prediction (non-blocking)
        try:
            self.feedback.log_prediction(
                request_id=request_id,
                image_key=image_key,
                disease=diagnosis.disease_name,
                confidence=diagnosis.confidence,
                treatment=treatment_en,
                language=language,
                inference_time_ms=diagnosis.inference_time_ms,
            )
        except Exception as e:
            logger.warning(f"Prediction logging failed (non-critical): {e}")

        # Build response
        total_time_ms = (time.monotonic() - start_time) * 1000

        response = {
            "request_id": request_id,
            "disease": {
                "name": diagnosis.disease_name,
                "crop": self._get_crop(diagnosis.disease_name),
                "severity": self._get_severity(diagnosis.disease_name),
                "confidence": round(diagnosis.confidence, 3),
                "confidence_level": diagnosis.confidence_level,
            },
            "treatment": {
                "english": treatment_en,
                "translated": translation["translated_text"],
                "language": translation["language"],
            },
            "audio": {
                "base64": audio_result["audio_base64"],
                "format": audio_result.get("format", "wav"),
                "sample_rate": audio_result.get("sample_rate", 22050),
            } if audio_result and audio_result.get("audio_base64") else None,
            "metadata": self._build_metadata(request_id, start_time, diagnosis),
        }

        logger.info(
            f"Pipeline complete: {diagnosis.disease_name} "
            f"(confidence={diagnosis.confidence:.2f}, total={total_time_ms:.0f}ms)"
        )

        return response

    def _should_use_bedrock(self, diagnosis: DiagnosisResult) -> bool:
        if diagnosis.confidence < 0.6:
            return True
        if len(diagnosis.treatment_english) < 50:
            return True
        if "unknown" in diagnosis.disease_name.lower():
            return True
        return False

    def _get_crop(self, disease_name: str) -> str:
        with open("configs/model_config.yaml") as f:
            config = yaml.safe_load(f)
        for cls in config["disease_classes"].values():
            if cls["name"] == disease_name:
                return cls["crop"]
        return "Unknown"

    def _get_severity(self, disease_name: str) -> str:
        with open("configs/model_config.yaml") as f:
            config = yaml.safe_load(f)
        for cls in config["disease_classes"].values():
            if cls["name"] == disease_name:
                return cls["severity"]
        return "unknown"

    def _build_metadata(self, request_id: str, start_time: float,
                        diagnosis: DiagnosisResult | None = None) -> dict:
        from datetime import datetime, timezone
        total_ms = (time.monotonic() - start_time) * 1000
        return {
            "request_id": request_id,
            "model_version": "1.1.0",
            "inference_time_ms": round(diagnosis.inference_time_ms if diagnosis else 0, 1),
            "total_time_ms": round(total_ms, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _error_response(self, request_id: str, code: str, message: str) -> dict:
        return {
            "request_id": request_id,
            "error": code,
            "message": message,
        }
