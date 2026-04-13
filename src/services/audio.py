"""
audio.py
--------
Text-to-speech using Amazon Polly.
Returns raw MP3 bytes — caller decides how to deliver (stream, S3, etc.)
"""

import logging
from typing import Literal

import boto3

logger = logging.getLogger(__name__)

POLLY_REGION = "us-east-1"

VOICE_MAP = {
    "en": "Joanna",
    "hi": "Aditi",
}


def text_to_speech(text: str, language: str = "en") -> bytes:
    """
    Convert text to MP3 audio bytes using Amazon Polly.

    Args:
        text    : plain text to speak (no markdown)
        language: "en" or "hi"

    Returns:
        MP3 audio bytes
    """
    lang_code = language if language in VOICE_MAP else "en"
    voice_id  = VOICE_MAP[lang_code]

    polly = boto3.client("polly", region_name=POLLY_REGION)
    resp  = polly.synthesize_speech(
        Text        =text,
        OutputFormat="mp3",
        VoiceId     =voice_id,
        Engine      ="standard",
    )
    audio_bytes = resp["AudioStream"].read()
    logger.info(f"TTS generated {len(audio_bytes)} bytes (lang={lang_code}, voice={voice_id})")
    return audio_bytes
