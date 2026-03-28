"""
Fine-tune Florence-2 on PlantVillage dataset using LoRA.

Usage:
    python training/fine_tune_florence.py
    python training/fine_tune_florence.py --config training/training_config.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class CropDiseaseDataset(Dataset):
    """Florence-2 compatible dataset for crop disease diagnosis."""

    def __init__(self, jsonl_path: str, processor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length
        self.entries = []

        with open(jsonl_path) as f:
            for line in f:
                self.entries.append(json.loads(line.strip()))

        logger.info(f"Loaded {len(self.entries)} entries from {jsonl_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        image = Image.open(entry["image_path"]).convert("RGB")
        prefix = entry["prefix"]  # "<CROP_DISEASE>"
        suffix = entry["suffix"]  # Disease name + treatment text

        # Process inputs
        inputs = self.processor(
            text=prefix,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Process labels
        labels = self.processor.tokenizer(
            suffix,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Squeeze batch dimension
        input_ids = inputs["input_ids"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)

        # Set padding tokens to -100 so they're ignored in loss
        label_ids[label_ids == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "labels": label_ids,
        }


def collate_fn(batch):
    """Custom collate for variable-length sequences."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def load_model_and_processor(config: dict):
    """Load Florence-2 with LoRA adapters."""
    model_name = config["model"]["name"]
    revision = config["model"]["revision"]

    logger.info(f"Loading model: {model_name} (revision: {revision})")

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        revision=revision,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        revision=revision,
        torch_dtype=torch.float16 if config["training"]["fp16"] else torch.float32,
    )

    # Freeze vision encoder
    if config["training"]["freeze_vision_encoder"]:
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        logger.info("Vision encoder frozen")

    # Apply LoRA
    lora_cfg = config["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def train(config: dict):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load model
    model, processor = load_model_and_processor(config)
    model.to(device)

    # Load datasets
    train_dataset = CropDiseaseDataset(
        "data/processed/train.jsonl",
        processor,
        max_length=config["training"]["max_length"],
    )
    val_dataset = CropDiseaseDataset(
        "data/processed/val.jsonl",
        processor,
        max_length=config["training"]["max_length"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["training"]["dataloader_num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["training"]["dataloader_num_workers"],
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler
    num_training_steps = len(train_loader) * config["training"]["epochs"]
    warmup_steps = int(num_training_steps * config["training"]["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    grad_accum = config["training"]["gradient_accumulation_steps"]

    logger.info(f"Starting training for {config['training']['epochs']} epochs")
    logger.info(f"Training samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Batch size: {config['training']['batch_size']}, Grad accum: {grad_accum}")
    logger.info(f"Effective batch size: {config['training']['batch_size'] * grad_accum}")

    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += outputs.loss.item()
            num_batches += 1

            if (step + 1) % config["training"]["logging_steps"] == 0:
                avg_loss = epoch_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch + 1} Step {step + 1}/{len(train_loader)} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                val_batches += 1

        avg_train_loss = epoch_loss / num_batches
        avg_val_loss = val_loss / val_batches

        logger.info(
            f"Epoch {epoch + 1}/{config['training']['epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_dir = Path(config["paths"]["best_model"])
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            logger.info(f"Saved best model (val_loss={avg_val_loss:.4f}) to {best_dir}")

        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            ckpt_dir = output_dir / f"checkpoint-epoch{epoch + 1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            logger.info(f"Saved checkpoint to {ckpt_dir}")

    logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {config['paths']['best_model']}")
    logger.info(f"Next step: python training/evaluate_model.py")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Florence-2 on crop disease data")
    parser.add_argument("--config", default="training/training_config.yaml", help="Config path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
