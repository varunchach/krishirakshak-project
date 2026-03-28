"""S3 image storage service."""

import logging
import uuid
from datetime import datetime

import boto3
import yaml

logger = logging.getLogger(__name__)


class S3Service:
    """Handle image uploads to S3."""

    def __init__(self, config_path: str = "configs/app_config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.bucket = config["storage"]["s3_bucket"]
        self.prefix = config["storage"]["s3_prefix"]
        self.client = boto3.client("s3")

    def upload_image(self, image_bytes: bytes, request_id: str) -> str:
        """Upload image and return S3 key."""
        timestamp = datetime.utcnow().strftime("%Y/%m/%d")
        key = f"{self.prefix}{timestamp}/{request_id}.jpg"

        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=image_bytes,
            ContentType="image/jpeg",
        )
        logger.info(f"Uploaded image: s3://{self.bucket}/{key}")
        return key
