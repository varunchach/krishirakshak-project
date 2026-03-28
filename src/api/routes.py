"""API routes for KrishiRakshak."""

import logging
import time
from datetime import datetime, timezone

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.schemas import (
    DiagnoseResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    LanguageInfo,
    LanguagesResponse,
)
from src.pipeline.diagnosis_pipeline import DiagnosisPipeline
from src.services.feedback_service import FeedbackService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")

# Lazy-init services
_pipeline: DiagnosisPipeline | None = None
_feedback: FeedbackService | None = None
_start_time = time.monotonic()


def get_pipeline() -> DiagnosisPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DiagnosisPipeline()
    return _pipeline


def get_feedback() -> FeedbackService:
    global _feedback
    if _feedback is None:
        _feedback = FeedbackService()
    return _feedback


@router.post("/diagnose", response_model=DiagnoseResponse, responses={400: {"model": ErrorResponse}})
async def diagnose(
    image: UploadFile = File(...),
    language: str = Form("en-IN"),
    include_audio: bool = Form(True),
):
    """Diagnose crop disease from a leaf image."""
    # Validate file type
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail={
            "error": "UNSUPPORTED_FORMAT",
            "message": f"Unsupported format: {image.content_type}. Use JPEG or PNG.",
        })

    # Read and validate size
    image_bytes = await image.read()
    max_size = 5 * 1024 * 1024  # 5MB
    if len(image_bytes) > max_size:
        raise HTTPException(status_code=400, detail={
            "error": "INVALID_IMAGE",
            "message": f"Image too large ({len(image_bytes) / 1e6:.1f}MB). Max: 5MB.",
        })

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail={
            "error": "INVALID_IMAGE",
            "message": "Empty image file.",
        })

    # Run pipeline
    pipeline = get_pipeline()
    result = pipeline.run(image_bytes, language=language, include_audio=include_audio)

    if "error" in result:
        raise HTTPException(status_code=503, detail=result)

    return result


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on a diagnosis."""
    feedback = get_feedback()
    feedback.submit_feedback(
        request_id=request.request_id,
        is_correct=request.is_correct,
        actual_disease=request.actual_disease,
        comment=request.comment,
    )
    return FeedbackResponse(request_id=request.request_id)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = int(time.monotonic() - _start_time)
    return HealthResponse(
        status="healthy",
        model_version="1.1.0",
        sagemaker_endpoint="active",
        sarvam_api="reachable",
        uptime_seconds=uptime,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/languages", response_model=LanguagesResponse)
async def list_languages():
    """List supported languages."""
    with open("configs/sarvam_config.yaml") as f:
        config = yaml.safe_load(f)

    languages = []
    tts_supported = set(config["tts"]["supported_languages"])

    for name, code in config["translation"]["supported_languages"].items():
        languages.append(LanguageInfo(
            code=code,
            name=name,
            tts_supported=code in tts_supported,
        ))

    # Add English
    languages.insert(0, LanguageInfo(code="en-IN", name="English", tts_supported=True))

    return LanguagesResponse(languages=languages)
