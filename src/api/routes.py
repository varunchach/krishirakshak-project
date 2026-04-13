"""
routes.py
---------
FastAPI route handlers for KrishiRakshak.

Endpoints:
  POST /v1/query      — text or Hindi query → agent response + audio
  POST /v1/diagnose   — image upload → disease diagnosis + treatment + audio
  POST /v1/ingest     — PDF upload → chunk + embed + store in FAISS
  POST /v1/feedback   — thumbs up/down on a response
  GET  /v1/health     — liveness + readiness check
"""

import logging
import os
import re
import time
import uuid
from datetime import date

import boto3
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from src.api.schemas        import (
    AgentResponse, FeedbackRequest, FeedbackResponse, HealthResponse,
    IngestResponse, QueryRequest,
)
from src.models.classifier  import get_backend_status, load_model, predict
from src.services.feedback_service import FeedbackService
from src.services.s3_service import S3Service

logger     = logging.getLogger(__name__)
router     = APIRouter(prefix="/v1")
_start     = time.monotonic()

# ── Lazy singletons ───────────────────────────────────────────────────────────
_agent     = None
_clf_model = None
_feedback  = FeedbackService()
_s3        = S3Service()

S3_BUCKET        = os.getenv("S3_BUCKET", "krishirakshak-assets")
CLASSIFIER_PATH  = os.getenv("CLASSIFIER_MODEL_PATH", "models_pkl/best_model.pth")
EMBEDDINGS_REGION = os.getenv("SAGEMAKER_REGION", "ap-south-1")
EMBEDDINGS_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "bge-m3-krishirakshak")


def build_graph():
    from src.agent.graph import build_graph as _build_graph

    return _build_graph()


def agent_run(*args, **kwargs):
    from src.agent.graph import run as _agent_run

    return _agent_run(*args, **kwargs)


def text_to_speech(*args, **kwargs):
    from src.services.audio import text_to_speech as _text_to_speech

    return _text_to_speech(*args, **kwargs)


def chunk_text(*args, **kwargs):
    try:
        from src.services.chunker import chunk_text as _chunk_text

        return _chunk_text(*args, **kwargs)
    except ImportError:
        text = args[0] if args else ""
        source = kwargs.get("source", "uploaded_doc")
        text = text.strip()
        if not text:
            return []
        return [
            {
                "text": text,
                "source": source,
                "chunk_index": 0,
                "language": detect_language(text),
                "ingested_at": str(date.today()),
                "disease_mentioned": [],
                "crop_mentioned": [],
                "pesticide_mentioned": [],
            }
        ]


def detect_language(*args, **kwargs):
    try:
        from src.services.chunker import detect_language as _detect_language

        return _detect_language(*args, **kwargs)
    except ImportError:
        text = args[0] if args else ""
        return "hi" if re.search(r"[\u0900-\u097F]", text) else "en"


def get_store():
    from src.services.retriever import get_store as _get_store

    return _get_store()


def _get_agent():
    global _agent
    if _agent is None:
        _agent = build_graph()
    return _agent


def _get_classifier():
    global _clf_model
    if _clf_model is None:
        _clf_model = load_model(CLASSIFIER_PATH)
    return _clf_model


# ── POST /v1/query ────────────────────────────────────────────────────────────
@router.post("/query", response_model=AgentResponse)
async def query(req: QueryRequest, request: Request):
    """Text or Hindi query — runs full ReAct agent pipeline."""
    start  = time.monotonic()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    answer = agent_run(
        user_input=req.query,
        session_id=req.session_id,
        agent=_get_agent(),
    )
    lang        = detect_language(answer)
    audio_bytes = text_to_speech(answer, lang)
    audio_url   = _s3.upload_audio(audio_bytes, key=f"audio/{req.session_id}/{request_id}.mp3")

    return AgentResponse(
        request_id =request_id,
        answer    =answer,
        session_id=req.session_id,
        language  =lang,
        audio_url =audio_url,
        latency_ms=round((time.monotonic() - start) * 1000, 1),
    )


# ── POST /v1/diagnose ─────────────────────────────────────────────────────────
@router.post("/diagnose", response_model=AgentResponse)
async def diagnose(
    request   : Request,
    image     : UploadFile = File(...),
    session_id: str        = Form("default"),
):
    """Image upload → EfficientNet diagnosis → RAG treatment + audio."""
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Unsupported image format. Use JPEG or PNG.")

    image_bytes = await image.read()
    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Max 5MB.")
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file.")

    start  = time.monotonic()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    result = predict(
        _get_classifier(),
        image_bytes=image_bytes,
        content_type=image.content_type or "image/jpeg",
    )
    image_key = _s3.upload_image(image_bytes, request_id=request_id)

    if result["low_conf"]:
        answer = (
            f"Confidence is too low ({result['confidence']}%). "
            "Please upload a clearer image or consult your local Krishi Vigyan Kendra."
        )
    else:
        answer = agent_run(
            user_input=f"What is the treatment for {result['disease']}?",
            session_id=session_id,
            agent=_get_agent(),
        )
        answer = f"Disease: {result['disease']} ({result['confidence']}%)\n\n{answer}"

    lang        = detect_language(answer)
    audio_bytes = text_to_speech(answer, lang)
    latency_ms  = round((time.monotonic() - start) * 1000, 1)
    audio_url   = _s3.upload_audio(audio_bytes, key=f"audio/{session_id}/{request_id}.mp3")
    _feedback.log_prediction(
        request_id=request_id,
        image_key=image_key,
        disease=result["disease"],
        confidence=result["confidence"],
        treatment=answer,
        language=lang,
        inference_time_ms=latency_ms,
    )

    return AgentResponse(
        request_id =request_id,
        answer    =answer,
        session_id=session_id,
        language  =lang,
        audio_url =audio_url,
        latency_ms=latency_ms,
    )


# ── POST /v1/ingest ───────────────────────────────────────────────────────────
@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    pdf    : UploadFile = File(...),
    source : str        = Form("uploaded_doc"),
):
    """PDF upload → extract → chunk → embed → store in FAISS index."""
    import fitz

    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")

    pdf_bytes = await pdf.read()
    tmp_path  = f"/tmp/{uuid.uuid4()}.pdf"
    with open(tmp_path, "wb") as f:
        f.write(pdf_bytes)

    doc  = fitz.open(tmp_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    os.remove(tmp_path)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    chunks = chunk_text(text, source=source)
    get_store().add_chunks(chunks)

    return IngestResponse(chunks_added=len(chunks), source=source, status="success")


# ── POST /v1/feedback ─────────────────────────────────────────────────────────
@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    """Record farmer feedback on a response."""
    _feedback.submit_feedback(
        request_id    =req.request_id,
        is_correct    =req.is_correct,
        actual_disease=req.actual_disease,
        comment       =req.comment,
    )
    return FeedbackResponse(request_id=req.request_id)


# ── GET /v1/health ────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
async def health():
    """Liveness + readiness check."""
    classifier = get_backend_status(_clf_model)
    embeddings = {
        "backend": "sagemaker",
        "endpoint": EMBEDDINGS_ENDPOINT,
        "region": EMBEDDINGS_REGION,
        "status": "unknown",
        "ready": False,
    }
    try:
        desc = boto3.client("sagemaker", region_name=EMBEDDINGS_REGION).describe_endpoint(
            EndpointName=EMBEDDINGS_ENDPOINT
        )
        embeddings["status"] = desc["EndpointStatus"]
        embeddings["ready"] = desc["EndpointStatus"] == "InService"
    except Exception as exc:
        embeddings["status"] = f"unreachable: {exc.__class__.__name__}"

    try:
        store = get_store()
        faiss_index_size = store.faiss_index.ntotal if store.faiss_index else 0
    except Exception as exc:
        logger.warning(f"Retriever health unavailable: {exc}")
        faiss_index_size = 0

    overall_status = "healthy" if classifier["ready"] and embeddings["ready"] else "degraded"
    return HealthResponse(
        status          =overall_status,
        classifier      =classifier,
        embeddings      =embeddings,
        faiss_index_size=faiss_index_size,
    )
