"""
embeddings.py
-------------
BGE-M3 embeddings via SageMaker endpoint.
Always use "query: " prefix for queries, "passage: " for documents.
Supports batched embedding to avoid endpoint timeouts.
"""

import json
import logging
import os
from typing import List

import boto3
import numpy as np
from botocore.config import Config as BotoConfig

logger = logging.getLogger(__name__)

SAGEMAKER_REGION   = os.getenv("SAGEMAKER_REGION", "ap-south-1")
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "bge-m3-krishirakshak")
BATCH_SIZE         = 8


def _get_runtime_client():
    return boto3.client(
        "sagemaker-runtime",
        region_name=SAGEMAKER_REGION,
        config=BotoConfig(read_timeout=300, connect_timeout=60),
    )


def get_embeddings(texts: List[str], prefix: str = "passage") -> np.ndarray:
    """
    Embed a list of texts using BGE-M3 on SageMaker.

    Args:
        texts  : list of strings to embed
        prefix : "passage" for documents, "query" for queries
    Returns:
        np.ndarray of shape (len(texts), 1024), float32, L2-normalized
    """
    runtime  = _get_runtime_client()
    prefixed = [f"{prefix}: {t}" for t in texts]
    all_vecs = []

    for i in range(0, len(prefixed), BATCH_SIZE):
        batch = prefixed[i:i + BATCH_SIZE]
        resp  = runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps({"inputs": batch, "normalize": True}),
        )
        raw    = json.loads(resp["Body"].read())
        vecs   = raw["embeddings"] if isinstance(raw, dict) else raw
        all_vecs.append(np.array(vecs, dtype="float32"))
        logger.debug(f"Embedded batch {i // BATCH_SIZE + 1} ({len(batch)} texts)")

    result = np.vstack(all_vecs)
    logger.info(f"Embedded {len(texts)} texts → shape {result.shape}")
    return result
