"""Sarvam Bulbul Text-to-Speech Service."""

import logging
import time

import boto3
import httpx
import yaml

logger = logging.getLogger(__name__)


class SarvamTTSService:
    """Generate speech from text using Sarvam Bulbul API."""

    def __init__(self, config_path: str = "configs/sarvam_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.base_url = self.config["api"]["base_url"]
        self.timeout = self.config["api"]["timeout"]
        self.tts_config = self.config["tts"]
        self.api_key = self._load_api_key()

        self.headers = {
            "API-Subscription-Key": self.api_key,
            "Content-Type": "application/json",
        }
        logger.info("Sarvam TTS service initialized")

    def _load_api_key(self) -> str:
        """Load API key from AWS SSM Parameter Store."""
        import os
        param_name = self.config["api"]["ssm_parameter_name"]
        try:
            client = boto3.client("ssm")
            response = client.get_parameter(Name=param_name, WithDecryption=True)
            return response["Parameter"]["Value"]
        except Exception:
            return os.environ.get("SARVAM_API_KEY", "")

    def synthesize(self, text: str, language: str, speaker: str | None = None) -> dict:
        """
        Convert text to speech audio.

        Args:
            text: Text to speak (in target language)
            language: Language code (e.g., 'hi-IN')
            speaker: Voice name (default: from config)

        Returns:
            dict with audio_base64, format, sample_rate, latency_ms
        """
        supported = self.tts_config["supported_languages"]

        if language not in supported:
            fallback = self.tts_config.get("fallback_for_unsupported", "hi-IN")
            logger.warning(f"TTS not supported for {language}, falling back to {fallback}")
            language = fallback

        speaker = speaker or self.tts_config["default_speaker"]
        start_time = time.monotonic()

        try:
            payload = {
                "inputs": text[:5000],  # Bulbul limit
                "target_language_code": language,
                "speaker": speaker,
                "model": self.tts_config["model"],
                "pitch": self.tts_config["pitch"],
                "pace": self.tts_config["pace"],
                "loudness": self.tts_config["loudness"],
                "speech_sample_rate": self.tts_config["speech_sample_rate"],
                "enable_preprocessing": self.tts_config["enable_preprocessing"],
            }

            with httpx.Client(timeout=self.timeout + 5) as client:
                response = client.post(
                    f"{self.base_url}{self.tts_config['endpoint']}",
                    headers=self.headers,
                    json=payload,
                )
                response.raise_for_status()

            result = response.json()
            latency_ms = (time.monotonic() - start_time) * 1000

            audios = result.get("audios", [])
            audio_b64 = audios[0] if audios else None

            logger.info(f"TTS generated for {language} ({len(text)} chars, {latency_ms:.0f}ms)")

            return {
                "audio_base64": audio_b64,
                "format": "wav",
                "sample_rate": self.tts_config["speech_sample_rate"],
                "language": language,
                "speaker": speaker,
                "latency_ms": latency_ms,
            }

        except httpx.HTTPStatusError as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.error(f"Sarvam TTS API error: {e.response.status_code}")
            return {"audio_base64": None, "latency_ms": latency_ms, "error": str(e)}

        except Exception as e:
            logger.error(f"TTS failed: {e}", exc_info=True)
            return {"audio_base64": None, "latency_ms": 0, "error": str(e)}
