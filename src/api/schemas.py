"""Pydantic request/response models for KrishiRakshak API."""

from pydantic import BaseModel, Field


class DiseaseInfo(BaseModel):
    name: str
    crop: str = ""
    severity: str = ""
    confidence: float = Field(ge=0, le=1)
    confidence_level: str


class TreatmentInfo(BaseModel):
    english: str
    translated: str = ""
    language: str = "en-IN"


class AudioInfo(BaseModel):
    base64: str
    format: str = "wav"
    sample_rate: int = 22050


class MetadataInfo(BaseModel):
    request_id: str
    model_version: str
    inference_time_ms: float
    total_time_ms: float
    timestamp: str


class DiagnoseResponse(BaseModel):
    request_id: str
    disease: DiseaseInfo
    treatment: TreatmentInfo
    audio: AudioInfo | None = None
    metadata: MetadataInfo


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    message: str


class FeedbackRequest(BaseModel):
    request_id: str
    is_correct: bool
    actual_disease: str | None = None
    comment: str | None = None


class FeedbackResponse(BaseModel):
    status: str = "recorded"
    request_id: str
    message: str = "Thank you for your feedback. This helps improve our system."


class HealthResponse(BaseModel):
    status: str
    model_version: str
    sagemaker_endpoint: str
    sarvam_api: str
    uptime_seconds: int
    timestamp: str


class LanguageInfo(BaseModel):
    code: str
    name: str
    tts_supported: bool


class LanguagesResponse(BaseModel):
    languages: list[LanguageInfo]
