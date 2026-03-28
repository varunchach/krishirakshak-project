"""Amazon Bedrock fallback for treatment advice augmentation."""

import json
import logging
import time

import boto3
import yaml

logger = logging.getLogger(__name__)


class BedrockFallback:
    """Generate detailed treatment advice when Florence-2 output is insufficient."""

    def __init__(self, config_path: str = "configs/app_config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.model_id = config["model"]["bedrock_model_id"]
        self.max_tokens = config["model"]["bedrock_max_tokens"]
        self.region = config["model"]["bedrock_region"]
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

        logger.info(f"Bedrock fallback initialized (model: {self.model_id})")

    def generate_treatment(self, disease_name: str, crop: str = "") -> dict:
        """Generate detailed India-specific treatment advice."""
        prompt = (
            f"You are an agricultural expert advising Indian farmers. "
            f"A farmer's {crop} crop has been diagnosed with: {disease_name}. "
            f"Provide treatment advice in simple, practical language. Include:\n"
            f"1) Immediate action needed right now\n"
            f"2) Recommended pesticide available in India (include brand names like Dithane M-45, Bavistin, Confidor etc.)\n"
            f"3) Exact dosage and application method\n"
            f"4) Prevention tips for next season\n\n"
            f"Keep it under 200 words. Use simple language a farmer can understand."
        )

        start = time.monotonic()
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }),
            )

            result = json.loads(response["body"].read())
            advice = result["content"][0]["text"]
            latency = (time.monotonic() - start) * 1000

            logger.info(f"Bedrock advice generated ({len(advice)} chars, {latency:.0f}ms)")
            return {"treatment": advice, "latency_ms": latency, "source": "bedrock"}

        except Exception as e:
            logger.error(f"Bedrock failed: {e}")
            return {"treatment": "", "latency_ms": 0, "error": str(e)}
