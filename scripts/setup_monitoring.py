"""
setup_monitoring.py
-------------------
Creates CloudWatch dashboard + alarms for KrishiRakshak monitoring.
Run once: python scripts/setup_monitoring.py

Creates:
  - Dashboard : KrishiRakshak-Monitoring
  - Alarm     : RAG-Faithfulness-Drift  (fires when faithfulness < 0.5 for 3 datapoints)
  - Alarm     : Classifier-Confidence-Drift (fires when confidence < 60 for 3 datapoints)
"""

import json
import boto3

REGION    = "us-east-1"
NAMESPACE = "KrishiRakshak"

cw = boto3.client("cloudwatch", region_name=REGION)

# ── Dashboard ─────────────────────────────────────────────────────────────────
dashboard_body = {
    "widgets": [
        {
            "type": "text",
            "x": 0, "y": 0, "width": 24, "height": 2,
            "properties": {
                "markdown": "# KrishiRakshak — Model Monitoring\nReal-time RAG quality and classifier drift detection. Powered by Llama 3.1 8B inline judge."
            }
        },
        {
            "type": "metric",
            "x": 0, "y": 2, "width": 8, "height": 6,
            "properties": {
                "title": "RAG Faithfulness Score",
                "metrics": [[NAMESPACE, "Faithfulness"]],
                "view": "timeSeries",
                "stat": "Average",
                "period": 300,
                "yAxis": {"left": {"min": 0, "max": 1}},
                "annotations": {
                    "horizontal": [{"value": 0.5, "label": "Drift threshold", "color": "#ff6b6b"}]
                },
                "region": REGION,
            }
        },
        {
            "type": "metric",
            "x": 8, "y": 2, "width": 8, "height": 6,
            "properties": {
                "title": "RAG Relevance Score",
                "metrics": [[NAMESPACE, "Relevance"]],
                "view": "timeSeries",
                "stat": "Average",
                "period": 300,
                "yAxis": {"left": {"min": 0, "max": 1}},
                "annotations": {
                    "horizontal": [{"value": 0.5, "label": "Drift threshold", "color": "#ff6b6b"}]
                },
                "region": REGION,
            }
        },
        {
            "type": "metric",
            "x": 16, "y": 2, "width": 8, "height": 6,
            "properties": {
                "title": "Classifier Confidence (%)",
                "metrics": [[NAMESPACE, "ClassifierConfidence"]],
                "view": "timeSeries",
                "stat": "Average",
                "period": 300,
                "yAxis": {"left": {"min": 0, "max": 100}},
                "annotations": {
                    "horizontal": [{"value": 60, "label": "Drift threshold", "color": "#ff6b6b"}]
                },
                "region": REGION,
            }
        },
        {
            "type": "metric",
            "x": 0, "y": 8, "width": 8, "height": 6,
            "properties": {
                "title": "API Latency (ms)",
                "metrics": [
                    [NAMESPACE, "LatencyMs", {"stat": "p50", "label": "p50"}],
                    [NAMESPACE, "LatencyMs", {"stat": "p95", "label": "p95"}],
                ],
                "view": "timeSeries",
                "period": 300,
                "region": REGION,
            }
        },
        {
            "type": "metric",
            "x": 8, "y": 8, "width": 8, "height": 6,
            "properties": {
                "title": "Guardrail Block Rate",
                "metrics": [[NAMESPACE, "GuardrailBlocked", {"stat": "Sum", "label": "Blocked requests"}]],
                "view": "timeSeries",
                "period": 300,
                "region": REGION,
            }
        },
        {
            "type": "alarm",
            "x": 16, "y": 8, "width": 8, "height": 6,
            "properties": {
                "title": "Drift Alarms",
                "alarms": [
                    f"arn:aws:cloudwatch:{REGION}:593755927741:alarm:KrishiRakshak-RAG-Faithfulness-Drift",
                    f"arn:aws:cloudwatch:{REGION}:593755927741:alarm:KrishiRakshak-Classifier-Confidence-Drift",
                ]
            }
        },
    ]
}

print("Creating CloudWatch dashboard...")
cw.put_dashboard(
    DashboardName="KrishiRakshak-Monitoring",
    DashboardBody=json.dumps(dashboard_body),
)
print("Dashboard created: KrishiRakshak-Monitoring")

# ── Alarms ────────────────────────────────────────────────────────────────────
print("Creating RAG faithfulness drift alarm...")
cw.put_metric_alarm(
    AlarmName       ="KrishiRakshak-RAG-Faithfulness-Drift",
    AlarmDescription="RAG faithfulness score dropped below 0.5 — possible drift or knowledge base degradation",
    Namespace       =NAMESPACE,
    MetricName      ="Faithfulness",
    Statistic       ="Average",
    Period          =300,        # 5 min window
    EvaluationPeriods=3,         # 3 consecutive periods
    Threshold       =0.5,
    ComparisonOperator="LessThanThreshold",
    TreatMissingData="notBreaching",
)
print("Alarm created: KrishiRakshak-RAG-Faithfulness-Drift")

print("Creating classifier confidence drift alarm...")
cw.put_metric_alarm(
    AlarmName       ="KrishiRakshak-Classifier-Confidence-Drift",
    AlarmDescription="EfficientNet confidence dropped below 60% — possible input distribution shift",
    Namespace       =NAMESPACE,
    MetricName      ="ClassifierConfidence",
    Statistic       ="Average",
    Period          =300,
    EvaluationPeriods=3,
    Threshold       =60.0,
    ComparisonOperator="LessThanThreshold",
    TreatMissingData="notBreaching",
)
print("Alarm created: KrishiRakshak-Classifier-Confidence-Drift")

print("\nDone. View your dashboard at:")
print(f"https://console.aws.amazon.com/cloudwatch/home?region={REGION}#dashboards:name=KrishiRakshak-Monitoring")
