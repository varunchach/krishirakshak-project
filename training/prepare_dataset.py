"""
Prepare PlantVillage dataset for Florence-2 fine-tuning.

Converts image classification dataset into seq2seq format:
  Input:  image + "<CROP_DISEASE>" prompt
  Output: "{disease_name}. Treatment: {treatment_text}"
"""

import json
import os
import random
import shutil
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

# Paths
RAW_DATA_DIR = Path("data/raw/plantvillage")
PROCESSED_DIR = Path("data/processed")
TREATMENT_KB_PATH = Path("training/treatment_kb.json")
CONFIG_PATH = Path("training/training_config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_treatment_kb() -> dict:
    with open(TREATMENT_KB_PATH) as f:
        return json.load(f)["diseases"]


def map_folder_to_disease(folder_name: str) -> str:
    """Map PlantVillage folder names to disease names in treatment KB."""
    mapping = {
        "Apple___Apple_scab": "Apple Scab",
        "Apple___Black_rot": "Apple Black Rot",
        "Apple___Cedar_apple_rust": "Apple Cedar Rust",
        "Apple___healthy": "Apple Healthy",
        "Blueberry___healthy": "Blueberry Healthy",
        "Cherry_(including_sour)___Powdery_mildew": "Cherry Powdery Mildew",
        "Cherry_(including_sour)___healthy": "Cherry Healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn Cercospora Leaf Spot",
        "Corn_(maize)___Common_rust_": "Corn Common Rust",
        "Corn_(maize)___Northern_Leaf_Blight": "Corn Northern Leaf Blight",
        "Corn_(maize)___healthy": "Corn Healthy",
        "Grape___Black_rot": "Grape Black Rot",
        "Grape___Esca_(Black_Measles)": "Grape Esca (Black Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape Leaf Blight",
        "Grape___healthy": "Grape Healthy",
        "Orange___Haunglongbing_(Citrus_greening)": "Orange Citrus Greening",
        "Peach___Bacterial_spot": "Peach Bacterial Spot",
        "Peach___healthy": "Peach Healthy",
        "Pepper,_bell___Bacterial_spot": "Bell Pepper Bacterial Spot",
        "Pepper,_bell___healthy": "Bell Pepper Healthy",
        "Potato___Early_blight": "Potato Early Blight",
        "Potato___Late_blight": "Potato Late Blight",
        "Potato___healthy": "Potato Healthy",
        "Raspberry___healthy": "Raspberry Healthy",
        "Soybean___healthy": "Soybean Healthy",
        "Squash___Powdery_mildew": "Squash Powdery Mildew",
        "Strawberry___Leaf_scorch": "Strawberry Leaf Scorch",
        "Strawberry___healthy": "Strawberry Healthy",
        "Tomato___Bacterial_spot": "Tomato Bacterial Spot",
        "Tomato___Early_blight": "Tomato Early Blight",
        "Tomato___Late_blight": "Tomato Late Blight",
        "Tomato___Leaf_Mold": "Tomato Leaf Mold",
        "Tomato___Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato Spider Mites",
        "Tomato___Target_Spot": "Tomato Target Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
        "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
        "Tomato___healthy": "Tomato Healthy",
    }
    return mapping.get(folder_name, folder_name)


def format_treatment_text(disease_name: str, treatment_kb: dict) -> str:
    """Format treatment info into a structured text output for Florence-2."""
    if disease_name not in treatment_kb:
        return f"{disease_name}. Treatment information not available."

    info = treatment_kb[disease_name]
    treatment = info["treatment"]

    text = (
        f"{disease_name}. "
        f"Severity: {info['severity']}. "
        f"Symptoms: {info['symptoms']}. "
        f"Immediate Action: {treatment['immediate_action']} "
        f"Pesticide: {treatment['pesticide']}. "
        f"Application: {treatment['application']}. "
        f"Prevention: {treatment['prevention']}"
    )
    return text


def build_dataset(raw_dir: Path, treatment_kb: dict) -> list[dict]:
    """Scan PlantVillage folders and build dataset entries."""
    entries = []

    # Find the actual image directory (may be nested)
    search_dirs = [raw_dir]
    for sub in raw_dir.iterdir():
        if sub.is_dir():
            search_dirs.append(sub)
            for subsub in sub.iterdir():
                if subsub.is_dir():
                    search_dirs.append(subsub)

    # Find the level with disease folders
    image_root = None
    for d in search_dirs:
        subdirs = [x for x in d.iterdir() if x.is_dir()]
        if len(subdirs) >= 30:  # PlantVillage has 38 classes
            image_root = d
            break

    if image_root is None:
        raise FileNotFoundError(
            f"Could not find PlantVillage class folders in {raw_dir}. "
            f"Expected 38 subdirectories."
        )

    print(f"Found image root: {image_root}")

    for class_dir in sorted(image_root.iterdir()):
        if not class_dir.is_dir():
            continue

        disease_name = map_folder_to_disease(class_dir.name)
        treatment_text = format_treatment_text(disease_name, treatment_kb)

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.png"))

        for img_path in images:
            entries.append({
                "image_path": str(img_path),
                "prefix": "<CROP_DISEASE>",
                "suffix": treatment_text,
                "disease_name": disease_name,
                "class_folder": class_dir.name,
            })

        print(f"  {disease_name}: {len(images)} images")

    return entries


def split_and_save(entries: list[dict], config: dict):
    """Split into train/val/test and save as JSONL."""
    random.seed(42)

    train_ratio = config["dataset"]["train_split"]
    val_ratio = config["dataset"]["val_split"]

    # Stratified split by disease class
    labels = [e["disease_name"] for e in entries]

    train_entries, temp_entries, _, temp_labels = train_test_split(
        entries, labels, test_size=(1 - train_ratio), stratify=labels, random_state=42
    )

    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_entries, test_entries = train_test_split(
        temp_entries, test_size=(1 - val_ratio_adjusted), stratify=temp_labels, random_state=42
    )

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train_entries), ("val", val_entries), ("test", test_entries)]:
        output_path = PROCESSED_DIR / f"{split_name}.jsonl"
        with open(output_path, "w") as f:
            for entry in split_data:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved {split_name}: {len(split_data)} entries → {output_path}")

    # Save class mapping
    unique_classes = sorted(set(e["disease_name"] for e in entries))
    class_map = {i: name for i, name in enumerate(unique_classes)}
    with open(PROCESSED_DIR / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"Saved class map: {len(class_map)} classes")

    # Dataset stats
    print(f"\n=== Dataset Summary ===")
    print(f"Total:  {len(entries)}")
    print(f"Train:  {len(train_entries)} ({train_ratio * 100:.0f}%)")
    print(f"Val:    {len(val_entries)} ({val_ratio * 100:.0f}%)")
    print(f"Test:   {len(test_entries)} ({(1 - train_ratio - val_ratio) * 100:.0f}%)")
    print(f"Classes: {len(unique_classes)}")


def main():
    print("=== KrishiRakshak Dataset Preparation ===\n")

    config = load_config()
    treatment_kb = load_treatment_kb()

    print(f"Loaded treatment KB: {len(treatment_kb)} diseases\n")

    if not RAW_DATA_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DATA_DIR}")
        print("Run: bash scripts/download_dataset.sh")
        return

    entries = build_dataset(RAW_DATA_DIR, treatment_kb)
    print(f"\nTotal entries: {len(entries)}\n")

    split_and_save(entries, config)

    print("\nDone! Next step: python training/fine_tune_florence.py")


if __name__ == "__main__":
    main()
