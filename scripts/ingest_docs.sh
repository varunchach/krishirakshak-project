#!/bin/bash
# ingest_docs.sh
# Ingests all PDFs from docs/ into the FAISS knowledge base via the API.
# Run this after the API server is up.
# Usage: bash scripts/ingest_docs.sh [API_URL]

set -euo pipefail

API_URL="${1:-http://localhost:8000}"
DOCS_DIR="$(dirname "$0")/../docs"
SUCCESS=0
FAILED=0

echo "==> Ingesting PDFs from docs/ into knowledge base"
echo "    API: $API_URL"
echo ""

for pdf in "$DOCS_DIR"/*.pdf; do
  filename=$(basename "$pdf")
  source_name="${filename%.pdf}"

  echo -n "    Ingesting: $filename ... "

  response=$(curl -s -o /tmp/ingest_resp.json -w "%{http_code}" \
    -X POST "${API_URL}/v1/ingest" \
    -F "pdf=@${pdf};type=application/pdf" \
    -F "source=${source_name}")

  if [ "$response" = "200" ]; then
    chunks=$(python3 -c "import json,sys; d=json.load(open('/tmp/ingest_resp.json')); print(d['chunks_added'])" 2>/dev/null || echo "?")
    echo "OK ($chunks chunks)"
    SUCCESS=$((SUCCESS + 1))
  else
    echo "FAILED (HTTP $response)"
    cat /tmp/ingest_resp.json 2>/dev/null || true
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "==> Done. Success: $SUCCESS | Failed: $FAILED"
