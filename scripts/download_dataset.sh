#!/bin/bash
# Download PlantVillage dataset from Kaggle

set -euo pipefail

DATASET_DIR="data/raw/plantvillage"

echo "=== KrishiRakshak Dataset Download ==="

# Check kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: kaggle CLI not found. Install with: pip install kaggle"
    echo "Ensure ~/.kaggle/kaggle.json has your API key"
    exit 1
fi

mkdir -p "$DATASET_DIR"

echo "Downloading PlantVillage dataset..."
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p "$DATASET_DIR" --unzip

echo "Dataset downloaded to $DATASET_DIR"
echo "Total images: $(find "$DATASET_DIR" -name '*.jpg' -o -name '*.JPG' -o -name '*.png' | wc -l)"
echo "Classes: $(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo ""
echo "Done! Next step: python training/prepare_dataset.py"
