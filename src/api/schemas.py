"""
schemas.py
----------
Pydantic request/response models for the KrishiRakshak API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query          : str  = Field(..., description="User query in Hindi or English")
    session_id     : str  = Field("default", description="Unique session ID per farmer")
    generate_audio : bool = Field(False, description="Set True to get audio response")


class DiagnoseRequest(BaseModel):
    image_path : str           = Field(..., description="S3 URI or local path to uploaded image")
    session_id : str           = Field("default", description="Unique session ID per farmer")
    query      : Optional[str] = Field(None, description="Optional follow-up question about the image")


class IngestRequest(BaseModel):
    pdf_path   : str           = Field(..., description="S3 URI or local path to PDF")
    source_name: Optional[str] = Field(None, description="Human-readable document name")


class AgentResponse(BaseModel):
    request_id   : Optional[str]   = None
    answer       : str
    session_id   : str
    language     : str
    audio_url    : Optional[str]   = None
    latency_ms   : Optional[float] = None
    eval_metrics : Optional[dict]  = None  # faithfulness, answer_relevance, context_relevance, context_precision


class IngestResponse(BaseModel):
    chunks_added: int
    source      : str
    status      : str


class FeedbackRequest(BaseModel):
    request_id    : str
    is_correct    : bool
    actual_disease: Optional[str] = None
    comment       : Optional[str] = None


class FeedbackResponse(BaseModel):
    status    : str = "recorded"
    request_id: str
    message   : str = "Thank you for your feedback."


class DependencyStatus(BaseModel):
    backend : str
    endpoint: Optional[str] = None
    region  : Optional[str] = None
    status  : str
    ready   : bool


class HealthResponse(BaseModel):
    status          : str
    classifier      : DependencyStatus
    embeddings      : DependencyStatus
    faiss_index_size: int
    version         : str = "1.0.0"
