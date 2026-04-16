"""
embeddings.py
-------------
BGE-M3 embeddings via SageMaker endpoint.
Always use "query: " prefix for queries, "passage: " for documents.

Cold start fix: endpoint is warmed at startup and kept alive with a
background ping every 4 minutes — users never hit a cold container.
"""

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import boto3
import numpy as np
from botocore.config import Config as BotoConfig

logger = logging.getLogger(__name__)

SAGEMAKER_REGION   = os.getenv("SAGEMAKER_REGION", "ap-south-1")
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "bge-m3-krishirakshak")
BATCH_SIZE         = 32  # increased from 8 — BGE-M3 handles larger batches fine
EMBED_WORKERS      = 4   # concurrent SageMaker calls during ingestion
KEEP_WARM_INTERVAL = 240  # seconds — ping every 4 min (serverless timeout ~5 min)

# ── Singleton client ──────────────────────────────────────────────────────────
_sm_client = None


def _get_sm_client():
    global _sm_client
    if _sm_client is None:
        _sm_client = boto3.client(
            "sagemaker-runtime",
            region_name=SAGEMAKER_REGION,
            config=BotoConfig(read_timeout=120, connect_timeout=10),
        )
    return _sm_client


# ── Warm-up ───────────────────────────────────────────────────────────────────

def warm_endpoint():
    """
    Send a single dummy embedding to wake the SageMaker container.
    Call once at startup, then let the keep-warm thread take over.
    """
    try:
        logger.info(f"Warming BGE-M3 endpoint ({SAGEMAKER_ENDPOINT})...")
        _get_sm_client().invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps({"inputs": ["warm"], "normalize": True}),
        )
        logger.info("BGE-M3 endpoint warm and ready")
    except Exception as e:
        logger.warning(f"BGE-M3 warm-up failed: {e}")


def start_keep_warm():
    """
    Background thread that pings the endpoint every KEEP_WARM_INTERVAL seconds
    so the serverless container never goes cold between user requests.
    """
    def _loop():
        while True:
            time.sleep(KEEP_WARM_INTERVAL)
            try:
                _get_sm_client().invoke_endpoint(
                    EndpointName=SAGEMAKER_ENDPOINT,
                    ContentType="application/json",
                    Body=json.dumps({"inputs": ["ping"], "normalize": True}),
                )
                logger.debug("BGE-M3 keep-warm ping sent")
            except Exception as e:
                logger.warning(f"BGE-M3 keep-warm ping failed: {e}")

    t = threading.Thread(target=_loop, daemon=True, name="bge-m3-keep-warm")
    t.start()
    logger.info(f"BGE-M3 keep-warm thread started (interval={KEEP_WARM_INTERVAL}s)")


# ── Public API ────────────────────────────────────────────────────────────────

def _embed_batch(batch: List[str], batch_idx: int, total_batches: int) -> tuple[int, np.ndarray]:
    """Embed a single batch — called concurrently from get_embeddings."""
    t0   = time.time()
    resp = _get_sm_client().invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT,
        ContentType="application/json",
        Body=json.dumps({"inputs": batch, "normalize": True}),
    )
    raw      = json.loads(resp["Body"].read())
    vecs     = raw["embeddings"] if isinstance(raw, dict) else raw
    duration = round((time.time() - t0) * 1000)
    logger.info(
        f"BGE-M3 | batch {batch_idx + 1}/{total_batches} | "
        f"texts={len(batch)} | endpoint={SAGEMAKER_ENDPOINT} | duration={duration}ms"
    )
    return batch_idx, np.array(vecs, dtype="float32")


def get_embeddings(texts: List[str], prefix: str = "passage") -> np.ndarray:
    """
    Embed texts using BGE-M3 on SageMaker.

    Args:
        texts  : list of strings to embed
        prefix : "passage" for documents, "query" for queries
    Returns:
        np.ndarray of shape (len(texts), 1024), float32, L2-normalised

    Batches are sent concurrently (EMBED_WORKERS) to maximise throughput
    during large document ingestion.
    """
    prefixed = [f"{prefix}: {t}" for t in texts]
    batches  = [prefixed[i:i + BATCH_SIZE] for i in range(0, len(prefixed), BATCH_SIZE)]
    workers  = min(EMBED_WORKERS, len(batches))
    results  = [None] * len(batches)

    logger.info(
        f"BGE-M3 embedding started | texts={len(texts)} | "
        f"batches={len(batches)} | workers={workers} | prefix={prefix}"
    )
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_embed_batch, batch, idx, len(batches)): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            idx, vecs    = future.result()
            results[idx] = vecs

    result   = np.vstack(results)
    total_ms = round((time.time() - t0) * 1000)
    logger.info(
        f"BGE-M3 embedding done | texts={len(texts)} | "
        f"shape={result.shape} | total_duration={total_ms}ms"
    )
    return result
