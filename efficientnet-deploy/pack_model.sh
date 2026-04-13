#!/bin/bash
# Pack best_model.pth + inference.py into model.tar.gz for SageMaker
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEIGHTS="$REPO_ROOT/models_pkl/best_model.pth"

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: $WEIGHTS not found"
    exit 1
fi

STAGING=$(mktemp -d)
cp "$WEIGHTS" "$STAGING/best_model.pth"
cp "$SCRIPT_DIR/inference.py" "$STAGING/inference.py"

tar -czf "$SCRIPT_DIR/model.tar.gz" -C "$STAGING" .
rm -rf "$STAGING"

echo "Created: $SCRIPT_DIR/model.tar.gz ($(du -sh "$SCRIPT_DIR/model.tar.gz" | cut -f1))"
