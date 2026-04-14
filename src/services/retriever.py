"""
retriever.py
------------
Hybrid retrieval: Vector search (FAISS) + BM25 lexical search fused via RRF.
Index is built in-memory and can be persisted to / loaded from S3.

Usage:
    store = VectorStore()
    store.add_chunks(chunks)          # list of chunk dicts from chunker.py
    results = store.search(query, k=5)
"""

import io
import json
import logging
import os
import pickle
from typing import List, Dict, Any, Optional

import boto3
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.models.embeddings import get_embeddings

logger = logging.getLogger(__name__)

RRF_K = 60   # RRF constant — higher = smoother rank fusion


class VectorStore:
    def __init__(self):
        self.chunks      : List[Dict[str, Any]] = []
        self.faiss_index : Optional[faiss.Index] = None
        self.bm25        : Optional[BM25Okapi]   = None
        self._dim        : Optional[int]          = None

    # ── Ingest ────────────────────────────────────────────────────────────────

    def add_chunks(self, new_chunks: List[Dict[str, Any]]) -> None:
        """Embed and add chunks to FAISS + BM25 indexes."""
        if not new_chunks:
            return

        texts    = [c["text"] for c in new_chunks]
        new_vecs = get_embeddings(texts, prefix="passage")

        if self.faiss_index is None:
            self._dim        = new_vecs.shape[1]
            self.faiss_index = faiss.IndexFlatIP(self._dim)

        self.faiss_index.add(new_vecs)
        self.chunks.extend(new_chunks)

        # Rebuild BM25 on full corpus
        tokenized = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

        logger.info(f"Index now has {self.faiss_index.ntotal} vectors")

        # Persist to S3 so index survives container restarts
        bucket = os.getenv("FAISS_S3_BUCKET")
        key    = os.getenv("FAISS_S3_KEY", "faiss_index/store.pkl")
        if bucket:
            try:
                self.save_to_s3(bucket, key)
            except Exception as e:
                logger.warning(f"Could not persist FAISS index to S3: {e}")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid search: FAISS vector + BM25 lexical, fused via RRF.

        Returns list of dicts: {chunk, metadata, score}
        """
        if self.faiss_index is None or not self.chunks:
            logger.warning("Index is empty — returning no results")
            return []

        fetch_k = min(k * 2, len(self.chunks))

        # Vector search
        q_vec            = get_embeddings([query], prefix="query")
        vec_scores, idxs = self.faiss_index.search(q_vec, fetch_k)
        vec_results      = {
            idxs[0][i]: i + 1
            for i in range(len(idxs[0]))
            if idxs[0][i] != -1
        }

        # BM25 search
        bm25_scores  = self.bm25.get_scores(query.lower().split())
        bm25_ranked  = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_k]
        bm25_results = {bm25_ranked[i]: i + 1 for i in range(len(bm25_ranked))}

        # RRF fusion
        all_ids = set(vec_results) | set(bm25_results)
        rrf     = {
            idx: 1 / (RRF_K + vec_results.get(idx, fetch_k + 1))
                + 1 / (RRF_K + bm25_results.get(idx, fetch_k + 1))
            for idx in all_ids
        }
        top_ids = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:k]

        return [
            {
                "chunk"   : self.chunks[i]["text"],
                "metadata": self.chunks[i],
                "score"   : round(rrf[i], 4),
            }
            for i in top_ids
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_to_s3(self, bucket: str, key: str) -> None:
        """Serialize index + chunks to S3."""
        data = {
            "chunks"     : self.chunks,
            "faiss_bytes": faiss.serialize_index(self.faiss_index).tobytes() if self.faiss_index else None,
            "dim"        : self._dim,
        }
        body = pickle.dumps(data)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=body)
        logger.info(f"Index saved to s3://{bucket}/{key}")

    def load_from_s3(self, bucket: str, key: str) -> None:
        """Load index + chunks from S3."""
        obj  = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        data = pickle.loads(obj["Body"].read())

        self.chunks = data["chunks"]
        self._dim   = data["dim"]

        if data["faiss_bytes"]:
            arr              = np.frombuffer(data["faiss_bytes"], dtype=np.uint8)
            self.faiss_index = faiss.deserialize_index(arr)

        tokenized = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"Index loaded from s3://{bucket}/{key} ({len(self.chunks)} chunks)")


# ── Raw document store (for small docs — no FAISS needed) ─────────────────────

class RawDocStore:
    """Stores full text of small documents for direct context passing to LLM."""

    def __init__(self):
        self.docs: Dict[str, str] = {}  # source → full text

    def add(self, source: str, text: str) -> None:
        self.docs[source] = text
        logger.info(f"RawDocStore: stored '{source}' ({len(text)} chars)")

    def get_all_text(self) -> str:
        """Concatenate all stored documents."""
        return "\n\n---\n\n".join(self.docs.values())

    def is_empty(self) -> bool:
        return len(self.docs) == 0

    def save_to_s3(self, bucket: str, key: str = "faiss_index/raw_docs.json") -> None:
        boto3.client("s3").put_object(
            Bucket=bucket, Key=key, Body=json.dumps(self.docs).encode()
        )
        logger.info(f"RawDocStore saved to s3://{bucket}/{key}")

    def load_from_s3(self, bucket: str, key: str = "faiss_index/raw_docs.json") -> None:
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        self.docs = json.loads(obj["Body"].read())
        logger.info(f"RawDocStore loaded {len(self.docs)} docs from s3://{bucket}/{key}")


# ── Module-level singleton ────────────────────────────────────────────────────
_store: Optional[VectorStore] = None
_raw_store: Optional[RawDocStore] = None


def get_raw_store() -> RawDocStore:
    global _raw_store
    if _raw_store is None:
        _raw_store = RawDocStore()
        bucket = os.getenv("FAISS_S3_BUCKET")
        if bucket:
            try:
                _raw_store.load_from_s3(bucket)
            except Exception as e:
                logger.warning(f"Could not load RawDocStore from S3 ({e}) — starting empty")
    return _raw_store


_LOCAL_INDEX_PATH = os.getenv("FAISS_LOCAL_PATH", "faiss_index/store.pkl")


def get_store() -> VectorStore:
    """
    Returns the singleton VectorStore.
    Load priority:
      1. Local disk (faiss_index/store.pkl) — baked into Docker image, instant load
      2. S3 — fallback for local dev when index not bundled
    """
    global _store
    if _store is None:
        _store = VectorStore()
        if os.path.exists(_LOCAL_INDEX_PATH):
            try:
                with open(_LOCAL_INDEX_PATH, "rb") as f:
                    data = pickle.load(f)
                _store.chunks = data["chunks"]
                _store._dim   = data["dim"]
                if data["faiss_bytes"]:
                    arr = np.frombuffer(data["faiss_bytes"], dtype=np.uint8)
                    _store.faiss_index = faiss.deserialize_index(arr)
                tokenized  = [c["text"].lower().split() for c in _store.chunks]
                _store.bm25 = BM25Okapi(tokenized)
                logger.info(f"FAISS index loaded from local disk ({len(_store.chunks)} chunks)")
            except Exception as e:
                logger.warning(f"Local index load failed ({e}) — falling back to S3")
                _store = VectorStore()

        if _store.faiss_index is None:
            bucket = os.getenv("FAISS_S3_BUCKET")
            key    = os.getenv("FAISS_S3_KEY", "faiss_index/store.pkl")
            if bucket:
                try:
                    _store.load_from_s3(bucket, key)
                    logger.info(f"FAISS index loaded from s3://{bucket}/{key}")
                except Exception as e:
                    logger.warning(f"Could not load FAISS index from S3 ({e}) — starting empty")
    return _store
