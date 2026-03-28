"""
Package fine-tuned Florence-2 and deploy to SageMaker.

Steps:
1. Create model.tar.gz with adapter weights + inference code
2. Upload to S3
3. Create SageMaker Model
4. Create/Update Endpoint
"""

import argparse
import json
import logging
import os
import shutil
import tarfile
from pathlib import Path

import boto3
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


INFERENCE_SCRIPT = '''
"""SageMaker inference handler for Florence-2."""

import base64
import io
import json
import os

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def model_fn(model_dir):
    """Load model for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(
        model_dir, trust_remote_code=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base-ft",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        cache_dir="/tmp/hf_cache",
    )

    model = PeftModel.from_pretrained(base_model, model_dir)
    model.to(device)
    model.eval()

    return {"model": model, "processor": processor, "device": device}


def input_fn(request_body, content_type):
    """Parse input request."""
    if content_type == "application/json":
        data = json.loads(request_body)
        image_bytes = base64.b64decode(data["image"])
        prompt = data.get("prompt", "<CROP_DISEASE>")
        return {"image_bytes": image_bytes, "prompt": prompt}
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model_dict):
    """Run inference."""
    model = model_dict["model"]
    processor = model_dict["processor"]
    device = model_dict["device"]

    image = Image.open(io.BytesIO(input_data["image_bytes"])).convert("RGB")

    inputs = processor(
        text=input_data["prompt"],
        images=image,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=3,
        )

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # Extract disease name (first sentence before period)
    parts = generated_text.split(". ", 1)
    disease_name = parts[0].strip()

    return {
        "generated_text": generated_text,
        "disease_name": disease_name,
        "confidence": 0.85,  # TODO: compute from logits
    }


def output_fn(prediction, accept):
    """Format output."""
    return json.dumps(prediction), "application/json"
'''


def package_model(config: dict):
    """Create model.tar.gz for SageMaker."""
    model_dir = Path(config["paths"]["best_model"])
    output_dir = Path(config["paths"]["sagemaker_model"])
    output_dir.mkdir(parents=True, exist_ok=True)

    staging_dir = output_dir / "staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()

    code_dir = staging_dir / "code"
    code_dir.mkdir()

    # Copy adapter weights and processor
    for f in model_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, staging_dir / f.name)
        elif f.is_dir():
            shutil.copytree(f, staging_dir / f.name)

    # Write inference script
    with open(code_dir / "inference.py", "w") as f:
        f.write(INFERENCE_SCRIPT)

    # Write requirements
    with open(code_dir / "requirements.txt", "w") as f:
        f.write("peft>=0.12.0\ntransformers>=4.44.0\ntorch>=2.4.0\ntimm>=1.0.0\neinops>=0.8.0\nPillow>=10.0.0\n")

    # Create tar.gz
    tar_path = output_dir / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for item in staging_dir.iterdir():
            tar.add(item, arcname=item.name)

    # Cleanup staging
    shutil.rmtree(staging_dir)

    size_mb = tar_path.stat().st_size / (1024 * 1024)
    logger.info(f"Model packaged: {tar_path} ({size_mb:.1f} MB)")
    return tar_path


def upload_to_s3(tar_path: Path, config: dict) -> str:
    """Upload model to S3."""
    s3_path = config["sagemaker"]["model_s3_path"]
    bucket = s3_path.replace("s3://", "").split("/")[0]
    key = "/".join(s3_path.replace("s3://", "").split("/")[1:]) + "model.tar.gz"

    s3 = boto3.client("s3")
    s3.upload_file(str(tar_path), bucket, key)

    full_s3_uri = f"s3://{bucket}/{key}"
    logger.info(f"Uploaded to {full_s3_uri}")
    return full_s3_uri


def deploy_endpoint(s3_uri: str, config: dict):
    """Create SageMaker model and endpoint."""
    sm = boto3.client("sagemaker")
    endpoint_name = config["sagemaker"]["endpoint_name"]
    model_name = f"{endpoint_name}-model"
    endpoint_config_name = f"{endpoint_name}-config"

    # Get execution role
    iam = boto3.client("iam")
    try:
        role = iam.get_role(RoleName="SageMakerExecutionRole")
        role_arn = role["Role"]["Arn"]
    except Exception:
        logger.error("SageMaker execution role not found. Run scripts/setup_aws.sh first.")
        return

    # Create model
    try:
        sm.delete_model(ModelName=model_name)
    except Exception:
        pass

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04",
            "ModelDataUrl": s3_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": s3_uri,
            },
        },
        ExecutionRoleArn=role_arn,
    )
    logger.info(f"Created model: {model_name}")

    # Create endpoint config (serverless)
    try:
        sm.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    except Exception:
        pass

    sm_config = config["sagemaker"]

    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": model_name,
            "ServerlessConfig": {
                "MemorySizeInMB": sm_config["serverless"]["memory_size_mb"],
                "MaxConcurrency": sm_config["serverless"]["max_concurrency"],
            },
        }],
    )
    logger.info(f"Created endpoint config: {endpoint_config_name}")

    # Create or update endpoint
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        logger.info(f"Updating endpoint: {endpoint_name}")
    except sm.exceptions.ClientError:
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        logger.info(f"Creating endpoint: {endpoint_name}")

    logger.info("Endpoint deployment initiated. This may take 5-10 minutes.")
    logger.info(f"Monitor at: https://console.aws.amazon.com/sagemaker/home#/endpoints/{endpoint_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/training_config.yaml")
    parser.add_argument("--skip-deploy", action="store_true", help="Only package, skip deploy")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tar_path = package_model(config)
    s3_uri = upload_to_s3(tar_path, config)

    if not args.skip_deploy:
        deploy_endpoint(s3_uri, config)


if __name__ == "__main__":
    main()
