"""Sarvam Mayura Translation Service."""

import logging
import time

import boto3
import httpx
import yaml

logger = logging.getLogger(__name__)


class SarvamTranslateService:
    """Translate text using Sarvam Mayura API."""

    def __init__(self, config_path: str = "configs/sarvam_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.base_url = self.config["api"]["base_url"]
        self.timeout = self.config["api"]["timeout"]
        self.api_key = self._load_api_key()

        self.headers = {
            "API-Subscription-Key": self.api_key,
            "Content-Type": "application/json",
        }
        logger.info("Sarvam Translate service initialized")

    def _load_api_key(self) -> str:
        """Load API key from AWS SSM Parameter Store."""
        import os
        param_name = self.config["api"]["ssm_parameter_name"]
        try:
            client = boto3.client("ssm")
            response = client.get_parameter(Name=param_name, WithDecryption=True)
            return response["Parameter"]["Value"]
        except Exception:
            key = os.environ.get("SARVAM_API_KEY", "")
            if not key:
                logger.warning("Sarvam API key not found in SSM Parameter Store or env")
            return key

    def translate(self, text: str, target_language: str) -> dict:
        """
        Translate English text to target Indian language.

        Args:
            text: English text to translate
            target_language: Language code (e.g., 'hi-IN', 'mr-IN')

        Returns:
            dict with translated_text and metadata
        """
        if target_language == "en-IN":
            return {"translated_text": text, "language": "en-IN", "latency_ms": 0}

        tr_config = self.config["translation"]
        supported = tr_config["supported_languages"]
        lang_names = {v: k for k, v in supported.items()}

        if target_language not in supported.values():
            logger.warning(f"Unsupported language: {target_language}, falling back to English")
            return {"translated_text": text, "language": "en-IN", "latency_ms": 0}

        start_time = time.monotonic()

        try:
            payload = {
                "input": text,
                "source_language_code": tr_config["source_language"],
                "target_language_code": target_language,
                "model": tr_config["model"],
                "mode": tr_config["mode"],
                "enable_preprocessing": tr_config["enable_preprocessing"],
            }

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}{tr_config['endpoint']}",
                    headers=self.headers,
                    json=payload,
                )
                response.raise_for_status()

            result = response.json()
            latency_ms = (time.monotonic() - start_time) * 1000

            translated = result.get("translated_text", text)
            logger.info(
                f"Translated to {lang_names.get(target_language, target_language)} "
                f"({len(text)} chars → {len(translated)} chars, {latency_ms:.0f}ms)"
            )

            return {
                "translated_text": translated,
                "language": target_language,
                "latency_ms": latency_ms,
            }

        except httpx.HTTPStatusError as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.error(f"Sarvam Translation API error: {e.response.status_code} - {e.response.text}")
            return {"translated_text": text, "language": "en-IN", "latency_ms": latency_ms, "error": str(e)}

        except Exception as e:
            logger.error(f"Translation failed: {e}", exc_info=True)
            return {"translated_text": text, "language": "en-IN", "latency_ms": 0, "error": str(e)}
