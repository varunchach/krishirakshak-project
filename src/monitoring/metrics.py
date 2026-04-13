"""Custom CloudWatch metrics for KrishiRakshak."""

import logging
import os

import boto3

logger = logging.getLogger(__name__)


class MetricsPublisher:
    """Publish custom metrics to CloudWatch."""

    def __init__(self):
        self.namespace = os.getenv("CLOUDWATCH_NAMESPACE", "KrishiRakshak")
        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.client = boto3.client("cloudwatch", region_name=region)

    def put_metric(self, name: str, value: float, unit: str = "None", dimensions: dict | None = None):
        try:
            dim_list = [{"Name": k, "Value": str(v)} for k, v in (dimensions or {}).items()]
            self.client.put_metric_data(
                Namespace=self.namespace,
                MetricData=[{
                    "MetricName": name,
                    "Value": value,
                    "Unit": unit,
                    "Dimensions": dim_list,
                }],
            )
        except Exception as e:
            logger.warning(f"Failed to publish metric {name}: {e}")

    def record_inference(self, latency_ms: float, confidence: float,
                         disease_class: str, model_version: str = "1.1.0"):
        self.put_metric("InferenceLatencyMs", latency_ms, "Milliseconds",
                        {"ModelVersion": model_version, "DiseaseClass": disease_class})
        self.put_metric("PredictionConfidence", confidence, "None",
                        {"DiseaseClass": disease_class})

    def record_translation(self, latency_ms: float, language: str):
        self.put_metric("TranslationLatencyMs", latency_ms, "Milliseconds",
                        {"TargetLanguage": language})

    def record_feedback(self, is_positive: bool):
        name = "FeedbackPositive" if is_positive else "FeedbackNegative"
        self.put_metric(name, 1, "Count")

    def record_request(self, endpoint: str, language: str):
        self.put_metric("RequestCount", 1, "Count",
                        {"Endpoint": endpoint, "Language": language})
