from __future__ import annotations

import argparse
import json
from math import sqrt

import boto3
from botocore.config import Config


REGION = "ap-south-1"
ENDPOINT_NAME = "krishirakshak-bge-m3"
READ_TIMEOUT_SECONDS = 310
EXPECTED_DIMENSION = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the BGE-M3 SageMaker endpoint.")
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--endpoint-name", default=ENDPOINT_NAME)
    return parser.parse_args()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm)


def invoke_endpoint(region: str, endpoint_name: str, texts: list[str]) -> dict:
    client = boto3.client(
        "sagemaker-runtime",
        region_name=region,
        config=Config(read_timeout=READ_TIMEOUT_SECONDS, connect_timeout=10, retries={"max_attempts": 5}),
    )
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps({"inputs": texts}).encode("utf-8"),
    )
    return json.loads(response["Body"].read())


def main() -> None:
    args = parse_args()
    texts = [
        "Rice leaves show yellowing and stunted growth due to nutrient stress.",
        "Les feuilles de riz jaunissent et la croissance est retardee a cause d un stress nutritionnel.",
        "The smartphone battery drains quickly during video calls.",
    ]
    result = invoke_endpoint(args.region, args.endpoint_name, texts)
    embeddings = result["embeddings"]
    dimensions = [len(embedding) for embedding in embeddings]
    same_topic_similarity = cosine_similarity(embeddings[0], embeddings[1])
    different_topic_similarity = cosine_similarity(embeddings[0], embeddings[2])

    if any(dimension != EXPECTED_DIMENSION for dimension in dimensions):
        raise RuntimeError(f"Unexpected embedding dimensions: {dimensions}")
    if same_topic_similarity <= different_topic_similarity:
        raise RuntimeError(
            "Cross-lingual similarity check failed: "
            f"same_topic={same_topic_similarity:.4f}, different_topic={different_topic_similarity:.4f}"
        )

    summary = {
        "endpoint_name": args.endpoint_name,
        "dimensions": dimensions,
        "same_topic_similarity": round(same_topic_similarity, 4),
        "different_topic_similarity": round(different_topic_similarity, 4),
        "check": "passed",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
