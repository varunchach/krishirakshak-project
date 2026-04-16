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

# ── Static treatment context for the 5 supported diseases ─────────────────────
# Eliminates FAISS from the diagnose path — instant dict lookup, no S3 needed.
_DISEASE_CONTEXT = {
    "Tomato Early Blight": (
        "Tomato Early Blight is caused by the fungus Alternaria solani. "
        "Symptoms: dark brown spots with concentric rings (target-board pattern) on older leaves, yellowing around spots. "
        "Treatment: Apply Mancozeb 75% WP (Dithane M-45, Indofil M-45) at 2.5 g per litre, or Chlorothalonil 75% WP (Kavach) at 2 g per litre. "
        "Spray every 7-10 days. Remove and destroy infected leaves. Avoid overhead irrigation. "
        "Prevention: Use certified disease-free seeds, maintain proper plant spacing, rotate with non-solanaceous crops next season."
    ),
    "Tomato Late Blight": (
        "Tomato Late Blight is caused by Phytophthora infestans. "
        "Symptoms: water-soaked greenish-black lesions on leaves, white cottony growth on leaf underside, rapid browning of stems and fruits. "
        "Treatment: Apply Metalaxyl + Mancozeb (Ridomil Gold MZ) at 2.5 g per litre, or Cymoxanil + Mancozeb (Curzate M8) at 2.5 g per litre. Spray every 7 days. "
        "Prevention: Avoid poorly drained areas, use resistant varieties, do not over-irrigate, destroy infected plant debris after harvest."
    ),
    "Potato Late Blight": (
        "Potato Late Blight is caused by Phytophthora infestans. "
        "Symptoms: water-soaked spots on leaves turning brown, white mold on undersides in humid conditions, tubers develop reddish-brown dry rot. "
        "Treatment: Apply Metalaxyl 8% + Mancozeb 64% WP (Ridomil Gold MZ) at 2.5 g per litre, or Copper oxychloride 50% WP (Blitox-50) at 3 g per litre. "
        "Prevention: Use certified disease-free seed tubers, plant in well-drained soil, apply earthing-up to protect tubers, avoid excess nitrogen, rotate crops next season."
    ),
    "Tomato Leaf Mold": (
        "Tomato Leaf Mold is caused by the fungus Passalora fulva. "
        "Symptoms: pale greenish-yellow spots on upper leaf surface, olive-green to grayish-purple mold on lower surface, leaves may curl and drop. "
        "Treatment: Apply Mancozeb 75% WP (Dithane M-45) at 2.5 g per litre, or Copper oxychloride (Blitox-50) at 3 g per litre. Improve ventilation. "
        "Prevention: Maintain good air circulation, keep humidity below 85%, avoid wetting foliage while irrigating, remove infected leaves promptly."
    ),
    "Corn Common Rust": (
        "Corn Common Rust is caused by the fungus Puccinia sorghi. "
        "Symptoms: small circular to elongated golden-brown to cinnamon-brown pustules on both leaf surfaces; pustules rupture releasing powdery rust-coloured spores. "
        "Treatment: Apply Propiconazole 25% EC (Tilt, Bumper) at 1 ml per litre of water, or Mancozeb 75% WP (Dithane M-45) at 2.5 g per litre. Spray when disease first appears. "
        "Prevention: Plant resistant hybrid varieties (consult local KVK), avoid excess nitrogen fertilizer, ensure good field drainage, maintain recommended plant spacing."
    ),
}


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


def _extract_pdf_text(pdf_bytes: bytes, filename: str) -> str:
    """
    Extract text from PDF — three-stage pipeline:
    1. fitz       — fast, handles most text-based PDFs
    2. pdfplumber — fallback for complex layouts / encoding edge cases
    3. AWS Textract — fallback for scanned / image-only PDFs (OCR)
    """
    import fitz
    import io

    # Open once — reused across all three stages to avoid parsing PDF bytes twice
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.warning(f"fitz could not open PDF: {e}")
        return ""

    # Stage 1 — fitz
    try:
        text = "\n".join(page.get_text() for page in doc)
    except Exception as e:
        logger.warning(f"fitz text extraction failed: {e}")
        text = ""

    if len(text.strip()) >= 100:
        doc.close()
        logger.info(f"Extracted {len(text)} chars via fitz")
        return text

    # Stage 2 — pdfplumber (better with complex layouts, tables, encodings)
    # Skip for large PDFs (> 1 MB) where fitz found nothing — almost certainly a
    # scanned/image PDF; pdfplumber won't help and is very slow on large files.
    if len(pdf_bytes) <= 1_000_000:
        logger.info(f"fitz got {len(text.strip())} chars — trying pdfplumber")
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages_text = [page.extract_text() or "" for page in pdf.pages]
            text = "\n".join(pages_text)
            logger.info(f"pdfplumber extracted {len(text)} chars")
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")

        if len(text.strip()) >= 100:
            doc.close()
            logger.info(f"Extracted {len(text)} chars via pdfplumber")
            return text
    else:
        logger.info(f"fitz got {len(text.strip())} chars on a {len(pdf_bytes)//1024}KB PDF — skipping pdfplumber, going straight to Textract")

    # Stage 3 — AWS Textract (scanned / image-only PDFs)
    # Reuses the already-open fitz doc — no second PDF parse needed.
    # Each page is rendered to PNG and sent to Textract synchronously.
    logger.info(f"Trying AWS Textract (OCR) on {len(doc)} pages")
    try:
        textract   = boto3.client("textract", region_name=os.getenv("SAGEMAKER_REGION", "ap-south-1"))
        pages_text = []

        for page_num, page in enumerate(doc):
            mat       = fitz.Matrix(150 / 72, 150 / 72)
            pix       = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")

            try:
                response   = textract.detect_document_text(Document={"Bytes": img_bytes})
                page_lines = [
                    block["Text"]
                    for block in response.get("Blocks", [])
                    if block["BlockType"] == "LINE"
                ]
                pages_text.append("\n".join(page_lines))
            except Exception as e:
                logger.warning(f"Textract failed on page {page_num + 1}: {e}")

        text = "\n".join(pages_text)
        logger.info(f"Textract extracted {len(text)} chars across {len(pages_text)} pages")
    except Exception as e:
        logger.error(f"AWS Textract stage failed: {e}")
    finally:
        doc.close()

    return text


def _get_classifier():
    global _clf_model
    if _clf_model is None:
        _clf_model = load_model(CLASSIFIER_PATH)
    return _clf_model


# ── POST /v1/query ────────────────────────────────────────────────────────────
@router.post("/query", response_model=AgentResponse)
async def query(req: QueryRequest, request: Request):
    """
    Text query — ReAct agent pipeline.

    Agent loop (graph.py):
      1. retriever_tool        — search FAISS knowledge base
      2. web_search_tool       — fallback if retrieval empty or insufficient
      3. rag_generator_tool    — generate final answer from context

    Small doc shortcut: if a small PDF was ingested (RawDocStore),
    pass full text directly to Claude — no embedding or agent overhead needed.
    """
    start      = time.monotonic()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # Guardrail check
    from src.services.guardrail import check as _guard, BLOCKED_RESPONSE
    from src.monitoring.monitor import log_rag_request
    is_safe, _ = _guard(req.query)
    if not is_safe:
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        log_rag_request(req.query, [], BLOCKED_RESPONSE, latency_ms, guardrail_blocked=True)
        return AgentResponse(
            request_id=request_id,
            answer=BLOCKED_RESPONSE,
            session_id=req.session_id,
            language="en",
            latency_ms=latency_ms,
        )

    from src.services.retriever import get_raw_store
    raw_store = get_raw_store()

    if not raw_store.is_empty():
        # Small doc path — full text directly to Claude, skip agent overhead
        from src.models.rag_generator import generate_direct as _generate_direct
        full_text = raw_store.get_all_text()
        answer    = _generate_direct(req.query, full_text)
        chunks    = [{"chunk": full_text[:2000], "score": 1.0}]  # synthetic chunk for eval
        logger.info("Query served via direct context (small doc)")

        # If Claude couldn't find the answer in the doc, fall back to ReAct agent (web search)
        _not_found_signals = ("not mentioned", "not found", "no information", "don't have",
                              "does not contain", "cannot find", "not available", "not provided")
        if any(s in answer.lower() for s in _not_found_signals):
            logger.info("Direct context answered 'not found' — falling back to ReAct agent (web search)")
            web_answer, chunks = agent_run(
                user_input=req.query,
                session_id=req.session_id,
                agent=_get_agent(),
            )
            answer = f"The ingested document did not contain this information. Based on web sources:\n\n{web_answer}"
            logger.info(f"Fallback ReAct agent returned {len(chunks)} chunks")
    else:
        # ReAct agent path — retriever → web search fallback → generate
        answer, chunks = agent_run(
            user_input=req.query,
            session_id=req.session_id,
            agent=_get_agent(),
        )
        logger.info(f"Query served via ReAct agent ({len(chunks)} context chunks retrieved)")

    latency_ms = round((time.monotonic() - start) * 1000, 1)

    # Compute eval metrics synchronously so they can be returned to the UI
    eval_metrics = None
    if chunks:
        try:
            from src.monitoring.monitor import judge_rag
            eval_metrics = judge_rag(req.query, chunks, answer)
        except Exception as e:
            logger.warning(f"Eval metrics failed: {e}")

    # Push to CloudWatch in background — never blocks the response
    import threading
    threading.Thread(
        target=log_rag_request,
        args=(req.query, chunks, answer, latency_ms),
        daemon=True,
    ).start()

    lang      = detect_language(answer)
    audio_url = None
    if req.generate_audio:
        audio_bytes = text_to_speech(answer, lang)
        audio_url   = _s3.upload_audio(audio_bytes, key=f"audio/{req.session_id}/{request_id}.mp3")

    return AgentResponse(
        request_id  =request_id,
        answer      =answer,
        session_id  =req.session_id,
        language    =lang,
        audio_url   =audio_url,
        latency_ms  =latency_ms,
        eval_metrics=eval_metrics,
    )


# ── POST /v1/diagnose ─────────────────────────────────────────────────────────
@router.post("/diagnose", response_model=AgentResponse)
async def diagnose(
    request       : Request,
    image         : UploadFile = File(...),
    session_id    : str        = Form("default"),
    generate_audio: bool       = Form(False),
):
    """Image upload → EfficientNet diagnosis → RAG treatment + audio."""
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Unsupported image format. Use JPEG or PNG.")

    image_bytes = await image.read()
    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Max 5MB.")
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file.")

    import threading
    start  = time.monotonic()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    result = predict(
        _get_classifier(),
        image_bytes=image_bytes,
        content_type=image.content_type or "image/jpeg",
    )

    if result["low_conf"]:
        answer = (
            f"Confidence is too low ({result['confidence']}%). "
            "Please upload a clearer image or consult your local Krishi Vigyan Kendra."
        )
    else:
        advice = _DISEASE_CONTEXT.get(
            result["disease"],
            "Please consult your local Krishi Vigyan Kendra for treatment advice.",
        )
        answer = f"Disease: {result['disease']} ({result['confidence']}%)\n\n{advice}"

    lang       = detect_language(answer)
    latency_ms = round((time.monotonic() - start) * 1000, 1)

    # S3 upload + feedback + monitoring — all fire-and-forget, don't block the response
    from src.monitoring.monitor import log_classifier_request
    def _background(img_bytes, res, ans, lang_, lat):
        img_key = _s3.upload_image(img_bytes, request_id=request_id)
        _feedback.log_prediction(
            request_id=request_id,
            image_key=img_key,
            disease=res["disease"],
            confidence=res["confidence"],
            treatment=ans,
            language=lang_,
            inference_time_ms=lat,
        )
        log_classifier_request(res["disease"], res["confidence"], lat)

    threading.Thread(
        target=_background,
        args=(image_bytes, result, answer, lang, latency_ms),
        daemon=True,
    ).start()

    audio_url  = None
    if generate_audio:
        audio_bytes = text_to_speech(answer, lang)
        audio_url   = _s3.upload_audio(audio_bytes, key=f"audio/{session_id}/{request_id}.mp3")

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
    logger.info(f"Ingest request: filename={pdf.filename!r} content_type={pdf.content_type!r}")

    if not (pdf.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")

    pdf_bytes = await pdf.read()
    logger.info(f"PDF size: {len(pdf_bytes)} bytes")

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Extract text — fitz → pdfplumber → Textract (scanned PDFs) ──
    text = _extract_pdf_text(pdf_bytes, pdf.filename)
    logger.info(f"Extracted {len(text)} chars from PDF")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    SMALL_DOC_THRESHOLD = 30_000  # chars (~7,500 tokens) — fits in Claude context

    if len(text) <= SMALL_DOC_THRESHOLD:
        # Small doc — store full text, skip chunking + embedding entirely
        from src.services.retriever import get_raw_store
        raw_store = get_raw_store()
        raw_store.add(source, text)
        bucket = os.getenv("FAISS_S3_BUCKET")
        if bucket:
            try:
                raw_store.save_to_s3(bucket)
            except Exception as e:
                logger.warning(f"Could not persist RawDocStore to S3: {e}")
        logger.info(
            f"Ingest complete [{source}] | strategy=direct_context | "
            f"chars={len(text)} | chunks=0 | embeddings=skipped (doc fits in LLM context)"
        )
        return IngestResponse(chunks_added=0, source=source, status="direct_context")
    else:
        # Large doc — chunk + embed into FAISS
        chunks = chunk_text(text, source=source)
        logger.info(f"Ingest [{source}] | chunking complete | chunks={len(chunks)}")
        get_store().add_chunks(chunks)
        logger.info(
            f"Ingest complete [{source}] | strategy=faiss+bm25 | "
            f"chars={len(text)} | chunks={len(chunks)} | embeddings=sagemaker bge-m3"
        )
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


# ── Health check cache — SageMaker DescribeEndpoint takes ~1.5s, ALB fires
# every 30s from 5 nodes with only 2 workers. Cache for 60s so health checks
# respond in <5ms and don't block workers during ingest/query requests.
_health_cache: dict = {}
_HEALTH_CACHE_TTL = 60  # seconds


# ── GET /v1/health ────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
async def health():
    """Liveness + readiness check."""
    now = time.monotonic()

    # Return cached result if fresh
    if _health_cache and (now - _health_cache["ts"]) < _HEALTH_CACHE_TTL:
        return HealthResponse(**_health_cache["data"])

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
    data = dict(
        status          =overall_status,
        classifier      =classifier,
        embeddings      =embeddings,
        faiss_index_size=faiss_index_size,
    )
    _health_cache["ts"]   = now
    _health_cache["data"] = data
    return HealthResponse(**data)
