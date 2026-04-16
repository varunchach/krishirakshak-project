"""
fetch_index.py
--------------
Downloads the FAISS index from S3 into faiss_index/ before Docker build.
Run this once before: bash scripts/deploy.sh

Usage:
    python scripts/fetch_index.py
"""

import os
import sys
import boto3
from pathlib import Path

BUCKET = os.getenv("FAISS_S3_BUCKET", "krishirakshak-assets")
KEY    = os.getenv("FAISS_S3_KEY",    "faiss_index/store.pkl")
OUT    = Path("faiss_index/store.pkl")

OUT.parent.mkdir(exist_ok=True)

print(f"Downloading s3://{BUCKET}/{KEY} → {OUT}")
try:
    boto3.client("s3").download_file(BUCKET, KEY, str(OUT))
    size_kb = OUT.stat().st_size // 1024
    print(f"Done. {size_kb} KB written to {OUT}")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
