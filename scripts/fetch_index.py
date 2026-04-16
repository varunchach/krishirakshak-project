"""
fetch_index.py
--------------
Downloads the FAISS index from S3 into faiss_index/ before Docker build.
If the index does not exist in S3 yet (first deploy), the directory is
created empty and the build continues — the API starts with an empty index
and populates it when users ingest documents.

Usage:
    python scripts/fetch_index.py
"""

import os
import sys
import boto3
from botocore.exceptions import ClientError
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
except ClientError as e:
    code = e.response["Error"]["Code"]
    if code in ("404", "NoSuchKey"):
        print(f"WARNING: FAISS index not found in S3 ({BUCKET}/{KEY}) — starting with empty index.")
        print("This is expected on first deploy. Ingest a PDF to populate the index.")
        sys.exit(0)
    else:
        print(f"ERROR: {e}")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
