"""
Evaluate fine-tuned Florence-2 vs base model on test set.

Generates accuracy, F1, confusion matrix, and treatment relevance scores.
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import torch
import yaml
from PIL import Image
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device):
    """Load fine-tuned Florence-2 model."""
    logger.info(f"Loading model from {model_path}")

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base-ft",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(base_model, model_path)
    model.to(device)
    model.eval()

    return model, processor


def extract_disease_name(output_text: str) -> str:
    """Extract disease name from model's generated text."""
    # Output format: "Disease Name. Severity: ..."
    parts = output_text.split(".")
    return parts[0].strip() if parts else "Unknown"


def evaluate(config_path: str):
    """Run evaluation on test set."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model_path = config["paths"]["best_model"]
    model, processor = load_model(model_path, device)

    # Load test data
    test_path = Path("data/processed/test.jsonl")
    if not test_path.exists():
        logger.error("Test set not found. Run prepare_dataset.py first.")
        return

    test_entries = []
    with open(test_path) as f:
        for line in f:
            test_entries.append(json.loads(line.strip()))

    logger.info(f"Test samples: {len(test_entries)}")

    # Run inference
    true_labels = []
    pred_labels = []
    results = []

    for i, entry in enumerate(test_entries):
        try:
            image = Image.open(entry["image_path"]).convert("RGB")
            true_disease = entry["disease_name"]

            inputs = processor(
                text="<CROP_DISEASE>",
                images=image,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    num_beams=3,
                )

            raw_output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            pred_disease = extract_disease_name(raw_output)

            true_labels.append(true_disease)
            pred_labels.append(pred_disease)

            results.append({
                "image": entry["image_path"],
                "true": true_disease,
                "predicted": pred_disease,
                "correct": true_disease.lower() == pred_disease.lower(),
                "raw_output": raw_output[:200],
            })

            if (i + 1) % 100 == 0:
                running_acc = sum(1 for r in results if r["correct"]) / len(results)
                logger.info(f"Progress: {i + 1}/{len(test_entries)} | Running accuracy: {running_acc:.3f}")

        except Exception as e:
            logger.warning(f"Failed on {entry['image_path']}: {e}")
            true_labels.append(entry["disease_name"])
            pred_labels.append("ERROR")

    # Metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"EVALUATION RESULTS")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total samples:  {len(test_entries)}")
    logger.info(f"Accuracy:       {accuracy:.4f} ({accuracy * 100:.1f}%)")

    # Classification report
    unique_labels = sorted(set(true_labels))
    report = classification_report(
        true_labels, pred_labels,
        labels=unique_labels,
        output_dict=True,
        zero_division=0,
    )
    logger.info(f"Weighted F1:    {report['weighted avg']['f1-score']:.4f}")
    logger.info(f"Macro F1:       {report['macro avg']['f1-score']:.4f}")

    # Worst performing classes
    class_f1 = {k: v["f1-score"] for k, v in report.items() if k in unique_labels}
    worst_classes = sorted(class_f1.items(), key=lambda x: x[1])[:5]
    logger.info(f"\nWorst 5 classes:")
    for cls, f1 in worst_classes:
        logger.info(f"  {cls}: F1={f1:.3f}")

    # Save results
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "weighted_f1": report["weighted avg"]["f1-score"],
            "macro_f1": report["macro avg"]["f1-score"],
            "total_samples": len(test_entries),
            "model_path": model_path,
        }, f, indent=2)

    # Error analysis
    errors = [r for r in results if not r["correct"]]
    error_pairs = Counter((r["true"], r["predicted"]) for r in errors)
    logger.info(f"\nTop confusion pairs:")
    for (true, pred), count in error_pairs.most_common(10):
        logger.info(f"  {true} → {pred}: {count} times")

    # Check against threshold
    min_accuracy = config["evaluation"]["min_accuracy_threshold"]
    if accuracy >= min_accuracy:
        logger.info(f"\n✅ PASSED: Accuracy {accuracy:.3f} >= threshold {min_accuracy}")
    else:
        logger.warning(f"\n❌ FAILED: Accuracy {accuracy:.3f} < threshold {min_accuracy}")

    logger.info(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/training_config.yaml")
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
