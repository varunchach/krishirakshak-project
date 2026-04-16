#!/usr/bin/env python3
"""
ingest_pipeline.py
------------------
Scheduled document ingestion pipeline for KrishiRakshak.

Scans s3://DOCS_S3_BUCKET/pipeline-docs/ for PDF files and ingests any that
are new or updated since the last run. Uses source-based deduplication so
re-uploading the same file never creates duplicate FAISS vectors.

Manifest (pipeline-docs/manifest.json in S3) tracks each file's ETag:
  - ETag unchanged  → skip (already up to date)
  - ETag changed    → remove old chunks, ingest fresh
  - File not in manifest → ingest as new

Usage:
    python scripts/ingest_pipeline.py                   # normal run
    python scripts/ingest_pipeline.py --force-reingest  # ignore manifest, re-ingest all

Environment variables:
    DOCS_S3_BUCKET     — bucket containing PDFs      (default: krishirakshak-assets-dev)
    DOCS_S3_PREFIX     — prefix to scan for PDFs     (default: pipeline-docs/)
    FAISS_S3_BUCKET    — bucket for FAISS index      (default: krishirakshak-assets-dev)
    FAISS_S3_KEY       — S3 key for FAISS index      (default: faiss_index/store.pkl)
    SAGEMAKER_REGION   — region for BGE-M3 endpoint  (default: ap-south-1)
    SAGEMAKER_ENDPOINT — SageMaker endpoint name     (default: bge-m3-krishirakshak)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# ── Make src/ importable when running from the project root ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Logging — structured, timestamped, stdout so GitHub Actions captures it ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("ingest_pipeline")

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_BUCKET  = os.getenv("DOCS_S3_BUCKET",    "krishirakshak-assets-dev")
DOCS_PREFIX  = os.getenv("DOCS_S3_PREFIX",    "pipeline-docs/")
FAISS_BUCKET = os.getenv("FAISS_S3_BUCKET",   "krishirakshak-assets-dev")
FAISS_KEY    = os.getenv("FAISS_S3_KEY",      "faiss_index/store.pkl")
MANIFEST_KEY = DOCS_PREFIX.rstrip("/") + "/manifest.json"

s3 = boto3.client("s3")


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _load_manifest() -> dict:
    """Load ingestion manifest from S3. Returns empty structure if not found."""
    try:
        obj  = s3.get_object(Bucket=DOCS_BUCKET, Key=MANIFEST_KEY)
        data = json.loads(obj["Body"].read())
        logger.info(
            f"Manifest loaded — {len(data.get('processed', {}))} previously processed file(s) | "
            f"last_run={data.get('last_run', 'never')}"
        )
        return data
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            logger.info("No manifest found in S3 — this is the first run, starting fresh")
            return {"processed": {}}
        raise


def _save_manifest(manifest: dict) -> None:
    """Persist updated manifest back to S3."""
    manifest["last_run"] = datetime.now(timezone.utc).isoformat()
    s3.put_object(
        Bucket=DOCS_BUCKET,
        Key=MANIFEST_KEY,
        Body=json.dumps(manifest, indent=2).encode(),
        ContentType="application/json",
    )
    logger.info(f"Manifest saved → s3://{DOCS_BUCKET}/{MANIFEST_KEY}")


# ── S3 listing ────────────────────────────────────────────────────────────────

def _list_pdfs() -> list:
    """Return all PDFs under DOCS_PREFIX as list of {key, etag, size_bytes}."""
    pdfs = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=DOCS_BUCKET, Prefix=DOCS_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".pdf"):
                pdfs.append({
                    "key"       : key,
                    "etag"      : obj["ETag"].strip('"'),
                    "size_bytes": obj["Size"],
                })
    logger.info(f"Found {len(pdfs)} PDF(s) in s3://{DOCS_BUCKET}/{DOCS_PREFIX}")
    for p in pdfs:
        logger.info(f"  {p['key'].split('/')[-1]} | {p['size_bytes']//1024}KB | etag={p['etag'][:8]}...")
    return pdfs


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_text(pdf_bytes: bytes, filename: str) -> str:
    """Extract text with fitz (PyMuPDF). Returns empty string on failure."""
    import fitz
    try:
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        logger.warning(f"fitz extraction failed for {filename}: {e}")
        return ""


# ── Per-document ingestion ────────────────────────────────────────────────────

def _process_one(pdf: dict, force: bool, manifest: dict, store) -> dict | None:
    """
    Ingest a single PDF into the FAISS store.
    Returns a manifest entry dict on success, None if skipped.
    """
    key      = pdf["key"]
    etag     = pdf["etag"]
    size     = pdf["size_bytes"]
    filename = key.split("/")[-1]
    existing = manifest.get("processed", {}).get(key, {})

    # ── Skip check ────────────────────────────────────────────────────────────
    if not force and existing.get("etag") == etag:
        logger.info(f"SKIP  {filename} — ETag unchanged, already ingested on {existing.get('processed_at', '?')[:10]}")
        return None

    action = "FORCE" if force else ("UPDATE" if existing else "NEW  ")
    logger.info("-" * 50)
    logger.info(f"{action} {filename} | {size//1024}KB | etag={etag[:8]}...")

    t0 = time.time()

    # ── [1/4] Download ────────────────────────────────────────────────────────
    logger.info(f"  [1/4] Downloading from s3://{DOCS_BUCKET}/{key}")
    dl_start  = time.time()
    pdf_bytes = s3.get_object(Bucket=DOCS_BUCKET, Key=key)["Body"].read()
    logger.info(f"  [1/4] Downloaded {len(pdf_bytes)//1024}KB in {round((time.time()-dl_start)*1000)}ms")

    # ── [2/4] Extract text ────────────────────────────────────────────────────
    logger.info(f"  [2/4] Extracting text via fitz (PyMuPDF)")
    ext_start = time.time()
    text      = _extract_text(pdf_bytes, filename)
    if not text.strip():
        logger.warning(f"  [2/4] No text extracted — skipping {filename}")
        return None
    logger.info(f"  [2/4] Extracted {len(text):,} chars in {round((time.time()-ext_start)*1000)}ms")

    # ── [3/4] Chunk (deduplication is automatic inside add_chunks) ───────────
    logger.info(f"  [3/4] Chunking with indic_rag_chunker")
    chunk_start = time.time()
    from src.services.chunker import chunk_text
    chunks = chunk_text(text, source=filename)
    logger.info(f"  [3/4] {len(chunks)} chunks in {round((time.time()-chunk_start)*1000)}ms")

    if not chunks:
        logger.warning(f"  [3/4] Chunker returned 0 chunks — skipping {filename}")
        return None

    # Log chunk sample for transparency
    for i, c in enumerate(chunks[:3], 1):
        preview = c["text"][:80].replace("\n", " ")
        logger.info(f"    Sample chunk #{i}: lang={c.get('language','?')} | {preview!r}...")
    if len(chunks) > 3:
        logger.info(f"    ... and {len(chunks)-3} more chunk(s)")

    # ── [4/4] Embed + Add to FAISS ────────────────────────────────────────────
    logger.info(f"  [4/4] Embedding {len(chunks)} chunks via BGE-M3 SageMaker")
    embed_start = time.time()
    store.add_chunks(chunks)
    embed_ms = round((time.time() - embed_start) * 1000)
    logger.info(f"  [4/4] Embedded and indexed in {embed_ms}ms | index_size={store.faiss_index.ntotal} vectors")

    total_ms = round((time.time() - t0) * 1000)
    logger.info(f"  DONE  {filename} | chunks={len(chunks)} | chars={len(text):,} | total={total_ms}ms")

    return {
        "etag"        : etag,
        "size_bytes"  : size,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "chunks_added": len(chunks),
        "chars"       : len(text),
        "strategy"    : "faiss+bm25",
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(force_reingest: bool = False):
    logger.info("=" * 60)
    logger.info("KrishiRakshak — Ingestion Pipeline START")
    logger.info(f"  Docs bucket : s3://{DOCS_BUCKET}/{DOCS_PREFIX}")
    logger.info(f"  FAISS index : s3://{FAISS_BUCKET}/{FAISS_KEY}")
    logger.info(f"  Force       : {force_reingest}")
    logger.info(f"  Run time    : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info("=" * 60)
    pipeline_start = time.time()

    # ── Warm BGE-M3 endpoint ──────────────────────────────────────────────────
    logger.info("Warming BGE-M3 SageMaker endpoint (avoids cold-start on first embed)...")
    from src.models.embeddings import warm_endpoint
    warm_endpoint()

    # ── Load existing FAISS index from S3 ─────────────────────────────────────
    from src.services.retriever import VectorStore
    store = VectorStore()
    try:
        store.load_from_s3(FAISS_BUCKET, FAISS_KEY)
        logger.info(f"Loaded existing FAISS index: {store.faiss_index.ntotal} vectors, {len(store.chunks)} chunks")
    except Exception as e:
        logger.info(f"No existing FAISS index ({e}) — will create a new one")

    # ── Load manifest ─────────────────────────────────────────────────────────
    manifest = _load_manifest()

    # ── Discover PDFs ─────────────────────────────────────────────────────────
    pdfs = _list_pdfs()
    if not pdfs:
        logger.info("No PDFs found in pipeline-docs/.")
        logger.info("Upload PDFs to s3://krishirakshak-assets-dev/pipeline-docs/ to ingest them.")
        logger.info("=" * 60)
        logger.info("Pipeline COMPLETE — nothing to process")
        logger.info("=" * 60)
        return

    # ── Process each PDF ──────────────────────────────────────────────────────
    processed_count = 0
    skipped_count   = 0
    failed_count    = 0

    for pdf in pdfs:
        try:
            entry = _process_one(pdf, force_reingest, manifest, store)
            if entry is None:
                skipped_count += 1
            else:
                manifest.setdefault("processed", {})[pdf["key"]] = entry
                processed_count += 1
        except Exception as e:
            logger.error(f"FAILED {pdf['key']}: {e}", exc_info=True)
            failed_count += 1

    # ── Persist FAISS index if anything changed ────────────────────────────────
    if processed_count > 0:
        logger.info("-" * 50)
        logger.info(f"Saving updated FAISS index → s3://{FAISS_BUCKET}/{FAISS_KEY}")
        save_start = time.time()
        store.save_to_s3(FAISS_BUCKET, FAISS_KEY)
        logger.info(f"FAISS index saved in {round((time.time()-save_start)*1000)}ms")

    # ── Persist manifest ───────────────────────────────────────────────────────
    _save_manifest(manifest)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_s = round(time.time() - pipeline_start)
    logger.info("=" * 60)
    logger.info("KrishiRakshak — Ingestion Pipeline COMPLETE")
    logger.info(f"  Processed : {processed_count} file(s)")
    logger.info(f"  Skipped   : {skipped_count} file(s) — already up-to-date")
    logger.info(f"  Failed    : {failed_count} file(s)")
    logger.info(f"  Index size: {store.faiss_index.ntotal if store.faiss_index else 0} vectors total")
    logger.info(f"  Duration  : {total_s}s")
    logger.info("=" * 60)

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KrishiRakshak scheduled ingestion pipeline")
    parser.add_argument(
        "--force-reingest",
        action="store_true",
        help="Re-ingest all docs even if ETag is unchanged (ignores manifest)",
    )
    args = parser.parse_args()
    run(force_reingest=args.force_reingest)
