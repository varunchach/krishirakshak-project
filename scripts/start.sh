#!/bin/bash
# start.sh - container entrypoint
# Production path: ECS invokes SageMaker endpoints for classifier + embeddings.
# Local path: set CLASSIFIER_BACKEND=local and mount model weights.

set -euo pipefail

echo "==> KrishiRakshak startup"
echo "    CLASSIFIER_BACKEND            : ${CLASSIFIER_BACKEND:-local}"
echo "    CLASSIFIER_SAGEMAKER_ENDPOINT : ${CLASSIFIER_SAGEMAKER_ENDPOINT:-<not set>}"
echo "    CLASSIFIER_SAGEMAKER_REGION   : ${CLASSIFIER_SAGEMAKER_REGION:-<not set>}"
echo "    EMBEDDINGS_SAGEMAKER_ENDPOINT : ${SAGEMAKER_ENDPOINT:-<not set>}"
echo "    FAISS_S3_BUCKET               : ${FAISS_S3_BUCKET:-<not set>}"

# FAISS index loads lazily through get_store() on first ingest/query request.
echo "==> Starting uvicorn (workers=${WORKERS:-2})..."
exec uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers "${WORKERS:-2}"
