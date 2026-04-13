"""S3 storage service — images and audio."""

import logging
import os
from datetime import datetime

import boto3

logger = logging.getLogger(__name__)

S3_BUCKET = os.getenv("S3_BUCKET", "krishirakshak-assets")
S3_PREFIX = os.getenv("S3_PREFIX", "images/")


class S3Service:
    """Handle uploads to S3 for images and audio."""

    def __init__(self):
        self.bucket = S3_BUCKET
        self.prefix = S3_PREFIX
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

    def upload_audio(self, audio_bytes: bytes, key: str) -> str:
        """
        Upload MP3 audio bytes to S3 and return a presigned URL (1 hour TTL).

        Args:
            audio_bytes : raw MP3 bytes from Polly
            key         : S3 key e.g. "audio/session-id/uuid.mp3"

        Returns:
            Presigned URL string
        """
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=audio_bytes,
            ContentType="audio/mpeg",
        )
        url = self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=3600,
        )
        logger.info(f"Uploaded audio: s3://{self.bucket}/{key}")
        return url
