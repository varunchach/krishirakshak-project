from __future__ import annotations

import argparse
import json
import mimetypes
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


ROLE_ARN = "arn:aws:iam::593755927741:role/service-role/AmazonSageMaker-ExecutionRole-20250503T142268"
REGION = "ap-south-1"
BUCKET = "sagemaker-bge-m3-593755927741"
ENDPOINT_NAME = "krishirakshak-bge-m3"
MODEL_NAME_PREFIX = "krishirakshak-bge-m3"
ENDPOINT_CONFIG_PREFIX = "krishirakshak-bge-m3-config"
IMAGE_URI = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-inference:2.2.0-cpu-py310"
MODEL_ARCHIVE = Path(__file__).with_name("model.tar.gz")
TIMEOUT_SECONDS = 300
SERVERLESS_MEMORY_MB = 3072
SERVERLESS_MAX_CONCURRENCY = 5


def build_clients(region: str) -> tuple[boto3.client, boto3.client]:
    api_config = Config(read_timeout=TIMEOUT_SECONDS + 20, connect_timeout=10, retries={"max_attempts": 10})
    s3_client = boto3.client("s3", region_name=region, config=api_config)
    sm_client = boto3.client("sagemaker", region_name=region, config=api_config)
    return s3_client, sm_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy the BGE-M3 model to a SageMaker serverless endpoint.")
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--endpoint-name", default=ENDPOINT_NAME)
    parser.add_argument("--role-arn", default=ROLE_ARN)
    parser.add_argument("--model-archive", default=str(MODEL_ARCHIVE))
    parser.add_argument("--s3-prefix", default="artifacts")
    return parser.parse_args()


def ensure_bucket(s3_client, bucket: str, region: str) -> None:
    try:
        s3_client.head_bucket(Bucket=bucket)
        print(f"Bucket exists: s3://{bucket}")
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        params = {"Bucket": bucket}
        if region != "us-east-1":
            params["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3_client.create_bucket(**params)
        print(f"Created bucket: s3://{bucket}")


def upload_model_archive(s3_client, archive: Path, bucket: str, s3_prefix: str) -> str:
    if not archive.exists():
        raise FileNotFoundError(f"Model archive not found: {archive}")

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    key = f"{s3_prefix.rstrip('/')}/{timestamp}/model.tar.gz"
    content_type, _ = mimetypes.guess_type(archive.name)
    extra_args = {"ContentType": content_type or "application/gzip"}

    print(f"Uploading {archive} to s3://{bucket}/{key} ...")
    s3_client.upload_file(str(archive), bucket, key, ExtraArgs=extra_args)
    model_uri = f"s3://{bucket}/{key}"
    print(f"Uploaded model archive: {model_uri}")
    return model_uri


def create_model(sm_client, model_name: str, role_arn: str, image_uri: str, model_uri: str) -> None:
    print(f"Creating SageMaker model: {model_name}")
    sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "TS_DEFAULT_RESPONSE_TIMEOUT": str(TIMEOUT_SECONDS),
            },
        },
    )


def create_endpoint_config(sm_client, endpoint_config_name: str, model_name: str) -> None:
    print(f"Creating endpoint config: {endpoint_config_name}")
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": SERVERLESS_MEMORY_MB,
                    "MaxConcurrency": SERVERLESS_MAX_CONCURRENCY,
                },
            }
        ],
    )


def endpoint_exists(sm_client, endpoint_name: str) -> bool:
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ValidationException":
            return False
        raise


def wait_for_endpoint(sm_client, endpoint_name: str) -> dict:
    print(f"Waiting for endpoint {endpoint_name} to reach InService ...")
    while True:
        description = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = description["EndpointStatus"]
        print(f"  status={status}")
        if status == "InService":
            return description
        if status == "Failed":
            raise RuntimeError(description.get("FailureReason", "Endpoint update failed"))
        waiter = sm_client.get_waiter("endpoint_in_service")
        waiter.wait(
            EndpointName=endpoint_name,
            WaiterConfig={"Delay": 30, "MaxAttempts": 120},
        )


def deploy(sm_client, endpoint_name: str, endpoint_config_name: str) -> dict:
    if endpoint_exists(sm_client, endpoint_name):
        print(f"Updating endpoint: {endpoint_name}")
        sm_client.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
    else:
        print(f"Creating endpoint: {endpoint_name}")
        sm_client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
    return wait_for_endpoint(sm_client, endpoint_name)


def current_timestamp_suffix() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def main() -> None:
    args = parse_args()
    archive = Path(args.model_archive).resolve()
    suffix = current_timestamp_suffix()
    model_name = f"{MODEL_NAME_PREFIX}-{suffix}"
    endpoint_config_name = f"{ENDPOINT_CONFIG_PREFIX}-{suffix}"

    s3_client, sm_client = build_clients(args.region)
    ensure_bucket(s3_client, args.bucket, args.region)
    model_uri = upload_model_archive(s3_client, archive, args.bucket, args.s3_prefix)
    create_model(sm_client, model_name, args.role_arn, IMAGE_URI, model_uri)
    create_endpoint_config(sm_client, endpoint_config_name, model_name)
    endpoint_description = deploy(sm_client, args.endpoint_name, endpoint_config_name)

    summary = {
        "region": args.region,
        "bucket": args.bucket,
        "model_archive": str(archive),
        "model_data_url": model_uri,
        "model_name": model_name,
        "endpoint_config_name": endpoint_config_name,
        "endpoint_name": args.endpoint_name,
        "endpoint_status": endpoint_description["EndpointStatus"],
        "timeout_seconds": TIMEOUT_SECONDS,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
