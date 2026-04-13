"""SNS alerting for anomalies and threshold breaches."""

import json
import logging
import os

import boto3

logger = logging.getLogger(__name__)


class AlertService:
    """Send alerts via SNS for operational issues."""

    def __init__(self):
        self.topic_name = os.getenv("SNS_TOPIC_NAME", "krishirakshak-alerts")
        self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.client = boto3.client("sns", region_name=self.region)
        self._topic_arn = None

    @property
    def topic_arn(self) -> str:
        if self._topic_arn is None:
            try:
                resp = self.client.create_topic(Name=self.topic_name)
                self._topic_arn = resp["TopicArn"]
            except Exception as e:
                logger.error(f"Failed to get SNS topic: {e}")
                self._topic_arn = ""
        return self._topic_arn

    def send_alert(self, subject: str, message: str, severity: str = "WARNING"):
        """Send an alert notification."""
        if not self.topic_arn:
            logger.warning(f"SNS topic not available. Alert: {subject}")
            return

        try:
            full_message = json.dumps({
                "severity": severity,
                "service": "KrishiRakshak",
                "subject": subject,
                "details": message,
            }, indent=2)

            self.client.publish(
                TopicArn=self.topic_arn,
                Subject=f"[{severity}] KrishiRakshak: {subject}"[:100],
                Message=full_message,
            )
            logger.info(f"Alert sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def alert_low_confidence(self, disease: str, confidence: float, request_id: str):
        self.send_alert(
            subject="Low Confidence Prediction",
            message=f"Disease: {disease}, Confidence: {confidence:.2f}, Request: {request_id}",
            severity="WARNING",
        )

    def alert_high_error_rate(self, error_rate: float, window_minutes: int):
        self.send_alert(
            subject="High Error Rate Detected",
            message=f"Error rate: {error_rate:.1f}% over last {window_minutes} minutes",
            severity="CRITICAL",
        )

    def alert_drift_detected(self, alerts: list[str]):
        self.send_alert(
            subject="Data Drift Detected",
            message=f"Drift indicators: {', '.join(alerts)}",
            severity="WARNING",
        )
