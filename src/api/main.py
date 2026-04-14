"""
main.py
-------
FastAPI entry point for KrishiRakshak.
"""

import logging
import os
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import LoggingMiddleware
from src.api.routes     import router, _get_classifier

logging.basicConfig(
    level  =logging.INFO,
    format ="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

app = FastAPI(
    title      ="KrishiRakshak API",
    version    ="1.0.0",
    description="AI-powered crop disease diagnosis for Indian farmers",
    docs_url   ="/docs",
    redoc_url  ="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins    =["*"],
    allow_credentials=True,
    allow_methods    =["*"],
    allow_headers    =["*"],
)

app.add_middleware(LoggingMiddleware)
app.include_router(router)


def _preload():
    """Warm up all components so the first user request is fast."""
    logger = logging.getLogger(__name__)

    try:
        _get_classifier()
        logger.info("Startup preload: classifier ready")
    except Exception as e:
        logger.warning(f"Startup preload: classifier failed — {e}")

    try:
        from src.models.embeddings import warm_endpoint, start_keep_warm
        warm_endpoint()       # blocks until endpoint is warm (~cold start time, once)
        start_keep_warm()     # background thread keeps it warm forever after
    except Exception as e:
        logger.warning(f"Startup preload: BGE-M3 warm-up failed — {e}")

    try:
        from src.services.retriever import get_store, get_raw_store
        get_raw_store()
        get_store()
        logger.info("Startup preload: FAISS + raw store ready")
    except Exception as e:
        logger.warning(f"Startup preload: FAISS failed — {e}")

    # No warm-up needed for Textract — it is a managed AWS service (no local model)


@app.on_event("startup")
async def startup_event():
    threading.Thread(target=_preload, daemon=True, name="preload").start()


@app.get("/")
async def root():
    return {
        "service": "KrishiRakshak",
        "version": "1.0.0",
        "docs"   : "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host   =os.getenv("HOST", "0.0.0.0"),
        port   =int(os.getenv("PORT", 8000)),
        workers=int(os.getenv("WORKERS", 1)),
        reload =os.getenv("DEBUG", "false").lower() == "true",
    )
