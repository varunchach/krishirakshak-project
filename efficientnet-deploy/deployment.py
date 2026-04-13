"""
Deploy EfficientNet-B3 classifier to SageMaker serverless endpoint.

Usage:
    python deployment.py [--region ap-south-1] [--bucket my-bucket] \
                         [--endpoint-name krishirakshak-efficientnet-b3] \
                         [--role-arn arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole]

Before running:
    bash pack_model.sh          # creates model.tar.gz from best_model.pth + inference.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ROLE_ARN = "arn:aws:iam::593755927741:role/service-role/AmazonSageMaker-ExecutionRole-20250503T142268"
REGION = "ap-south-1"
BUCKET = "sagemaker-bge-m3-593755927741"          # reuse existing models bucket
ENDPOINT_NAME = "krishirakshak-efficientnet-b3"
MODEL_NAME_PREFIX = "krishirakshak-efficientnet-b3"
ENDPOINT_CONFIG_PREFIX = "krishirakshak-efficientnet-b3-config"
IMAGE_URI = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-inference:2.2.0-cpu-py310"
MODEL_ARCHIVE = Path(__file__).with_name("model.tar.gz")

SERVERLESS_MEMORY_MB = 3072
SERVERLESS_MAX_CONCURRENCY = 5


def _clients(region: str):
    cfg = Config(read_timeout=320, connect_timeout=10, retries={"max_attempts": 10})
    return (
        boto3.client("s3", region_name=region, config=cfg),
        boto3.client("sagemaker", region_name=region, config=cfg),
    )


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default=REGION)
    p.add_argument("--bucket", default=BUCKET)
    p.add_argument("--endpoint-name", default=ENDPOINT_NAME)
    p.add_argument("--role-arn", default=ROLE_ARN)
    p.add_argument("--model-archive", default=str(MODEL_ARCHIVE))
    p.add_argument("--s3-prefix", default="artifacts/efficientnet-b3")
    return p.parse_args()


def _upload_archive(s3, archive: Path, bucket: str, prefix: str) -> str:
    if not archive.exists():
        raise FileNotFoundError(f"model.tar.gz not found at {archive}. Run: bash pack_model.sh")
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    key = f"{prefix.rstrip('/')}/{ts}/model.tar.gz"
    print(f"Uploading {archive} ({archive.stat().st_size // 1024} KB) -> s3://{bucket}/{key}")
    s3.upload_file(str(archive), bucket, key, ExtraArgs={"ContentType": "application/gzip"})
    return f"s3://{bucket}/{key}"


def _endpoint_exists(sm, name: str) -> bool:
    try:
        sm.describe_endpoint(EndpointName=name)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            return False
        raise


def _wait(sm, name: str) -> str:
    print(f"Waiting for endpoint {name} ...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=name, WaiterConfig={"Delay": 30, "MaxAttempts": 120})
    status = sm.describe_endpoint(EndpointName=name)["EndpointStatus"]
    print(f"Endpoint status: {status}")
    return status


def main():
    args = _parse_args()
    archive = Path(args.model_archive).resolve()
    suffix = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    model_name = f"{MODEL_NAME_PREFIX}-{suffix}"
    config_name = f"{ENDPOINT_CONFIG_PREFIX}-{suffix}"

    s3, sm = _clients(args.region)

    model_uri = _upload_archive(s3, archive, args.bucket, args.s3_prefix)

    print(f"Creating SageMaker model: {model_name}")
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=args.role_arn,
        PrimaryContainer={
            "Image": IMAGE_URI,
            "ModelDataUrl": model_uri,
            "Environment": {"SAGEMAKER_PROGRAM": "inference.py"},
        },
    )

    print(f"Creating endpoint config: {config_name}")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "ServerlessConfig": {
                "MemorySizeInMB": SERVERLESS_MEMORY_MB,
                "MaxConcurrency": SERVERLESS_MAX_CONCURRENCY,
            },
        }],
    )

    if _endpoint_exists(sm, args.endpoint_name):
        print(f"Updating existing endpoint: {args.endpoint_name}")
        sm.update_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=config_name)
    else:
        print(f"Creating endpoint: {args.endpoint_name}")
        sm.create_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=config_name)

    status = _wait(sm, args.endpoint_name)

    print(json.dumps({
        "endpoint_name": args.endpoint_name,
        "endpoint_status": status,
        "model_data_url": model_uri,
        "region": args.region,
    }, indent=2))


if __name__ == "__main__":
    main()
