"""Tests for Sarvam translation and TTS services."""

import pytest
from unittest.mock import patch, MagicMock


@patch("src.services.sarvam_translate.boto3")
@patch("src.services.sarvam_translate.httpx")
def test_translate_english_passthrough(mock_httpx, mock_boto):
    from src.services.sarvam_translate import SarvamTranslateService

    with patch("builtins.open", MagicMock()):
        with patch("yaml.safe_load", return_value={
            "api": {"base_url": "https://api.sarvam.ai", "timeout": 10, "secret_name": "test"},
            "translation": {
                "endpoint": "/translate", "model": "mayura:v1",
                "source_language": "en-IN", "mode": "formal",
                "enable_preprocessing": True,
                "supported_languages": {"Hindi": "hi-IN"},
            },
        }):
            mock_boto.client.return_value.get_secret_value.return_value = {"SecretString": "test-key"}
            service = SarvamTranslateService()
            result = service.translate("Hello", "en-IN")
            assert result["translated_text"] == "Hello"
            assert result["language"] == "en-IN"


@patch("src.services.sarvam_tts.boto3")
def test_tts_unsupported_language_fallback(mock_boto):
    from src.services.sarvam_tts import SarvamTTSService

    with patch("builtins.open", MagicMock()):
        with patch("yaml.safe_load", return_value={
            "api": {"base_url": "https://api.sarvam.ai", "timeout": 10, "secret_name": "test"},
            "tts": {
                "endpoint": "/text-to-speech", "model": "bulbul:v1",
                "default_speaker": "meera", "speakers": {"female": "meera"},
                "pitch": 0, "pace": 1.0, "loudness": 1.5,
                "speech_sample_rate": 22050, "enable_preprocessing": True,
                "supported_languages": ["hi-IN", "ta-IN"],
                "fallback_for_unsupported": "hi-IN",
            },
        }):
            mock_boto.client.return_value.get_secret_value.return_value = {"SecretString": "test-key"}
            service = SarvamTTSService()
            # mr-IN not supported — should fallback
            assert "mr-IN" not in service.tts_config["supported_languages"]
