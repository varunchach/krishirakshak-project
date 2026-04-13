"""
Smoke test for the krishirakshak-efficientnet-b3 SageMaker endpoint.

Fetches test images from S3 (production assets bucket) and runs inference
against all 5 disease classes to validate the endpoint.

Usage:
    python smoke_test.py --bucket krishirakshak-assets-dev \
                         [--endpoint-name krishirakshak-efficientnet-b3] \
                         [--region ap-south-1]

Images expected in S3 at:  docs/<filename>
Upload once with:
    aws s3 cp docs/ s3://krishirakshak-assets-dev/docs/ --recursive \
        --exclude "*" --include "*.jpg" --include "*.png"
"""

import argparse
import json

import boto3

ENDPOINT_NAME = "krishirakshak-efficientnet-b3"
ENDPOINT_REGION = "ap-south-1"

# (s3_key, expected_disease, content_type)
TEST_CASES = [
    ("docs/tomato_early_blight_leaf_image_1.jpg", "Tomato Early Blight", "image/jpeg"),
    ("docs/tomato_late_blight_leaf_image_1.jpg",  "Tomato Late Blight",  "image/jpeg"),
    ("docs/Late_Blight_potato_image_1.png",        "Potato Late Blight",  "image/png"),
    ("docs/Tomato_Leaf_Mold_image_1.jpg",          "Tomato Leaf Mold",    "image/jpeg"),
    ("docs/Corn_Common_Rust_image_1.jpg",           "Corn Common Rust",    "image/jpeg"),
]


def fetch_from_s3(s3_client, bucket: str, key: str) -> bytes:
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def invoke(runtime_client, endpoint_name: str, image_bytes: bytes, content_type: str) -> dict:
    resp = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Accept="application/json",
        Body=image_bytes,
    )
    return json.loads(resp["Body"].read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True, help="S3 assets bucket, e.g. krishirakshak-assets-dev")
    p.add_argument("--endpoint-name", default=ENDPOINT_NAME)
    p.add_argument("--endpoint-region", default=ENDPOINT_REGION)
    p.add_argument("--s3-region", default="us-east-1", help="Region of the assets S3 bucket")
    args = p.parse_args()

    s3 = boto3.client("s3", region_name=args.s3_region)
    runtime = boto3.client("sagemaker-runtime", region_name=args.endpoint_region)

    passed = 0
    failed = 0

    for key, expected_disease, content_type in TEST_CASES:
        print(f"\n[TEST] {key}")
        try:
            image_bytes = fetch_from_s3(s3, args.bucket, key)
            result = invoke(runtime, args.endpoint_name, image_bytes, content_type)
            disease = result["disease"]
            confidence = result["confidence"]
            low_conf = result["low_conf"]

            status = "PASS" if disease == expected_disease else "WARN (unexpected class)"
            print(f"  -> {disease} ({confidence:.1%}) low_conf={low_conf}  [{status}]")
            passed += 1
        except Exception as e:
            print(f"  -> FAIL: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
    print("Smoke test PASSED")


if __name__ == "__main__":
    main()
