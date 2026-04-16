"""Feedback and prediction logging service (DynamoDB)."""

import logging
import os
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)

DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "krishirakshak-predictions-dev")


class FeedbackService:
    """Log predictions and farmer feedback to DynamoDB."""

    def __init__(self):
        self.table_name = DYNAMODB_TABLE
        self._table = None  # lazy — connect on first use, not at import time

    @property
    def table(self):
        if self._table is None:
            self._table = boto3.resource(
                "dynamodb",
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            ).Table(self.table_name)
        return self._table

    def log_prediction(self, request_id: str, image_key: str, disease: str,
                       confidence: float, treatment: str, language: str,
                       inference_time_ms: float):
        """Log a prediction for later feedback matching."""
        try:
            self.table.put_item(Item={
                "PK"              : f"REQUEST#{request_id}",
                "SK"              : "PREDICTION",
                "disease"         : disease,
                "confidence"      : str(confidence),
                "treatment_en"    : treatment[:1000],
                "language"        : language,
                "image_s3_key"    : image_key,
                "inference_time_ms": str(inference_time_ms),
                "timestamp"       : datetime.now(timezone.utc).isoformat(),
            })
            logger.info(f"Logged prediction for {request_id}")
        except Exception as e:
            # Don't fail the request if logging fails
            logger.error(f"Failed to log prediction: {e}")

    def submit_feedback(self, request_id: str, is_correct: bool,
                        actual_disease: str | None = None,
                        comment: str | None = None):
        """Record farmer feedback on a diagnosis."""
        try:
            self.table.put_item(Item={
                "PK"            : f"REQUEST#{request_id}",
                "SK"            : "FEEDBACK",
                "is_correct"    : is_correct,
                "actual_disease": actual_disease or "",
                "comment"       : comment or "",
                "submitted_at"  : datetime.now(timezone.utc).isoformat(),
            })
            logger.info(f"Recorded feedback for {request_id}: correct={is_correct}")
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            raise
