"""
monitor.py
----------
Inline RAG + classifier monitoring using Llama 3.1 8B as judge.
Pushes custom metrics to CloudWatch for drift detection.

Metrics published:
  KrishiRakshak/RAG/Faithfulness     — 0-1, is answer grounded in context?
  KrishiRakshak/RAG/Relevance        — 0-1, does answer address the query?
  KrishiRakshak/Classifier/Confidence — 0-100, EfficientNet confidence
  KrishiRakshak/API/GuardrailBlocked — 0 or 1 per request
  KrishiRakshak/API/LatencyMs        — end-to-end latency
"""

import json
import logging
import os
from typing import List, Dict, Any

import boto3

logger = logging.getLogger(__name__)

JUDGE_MODEL   = "us.meta.llama3-1-8b-instruct-v1:0"
BEDROCK_REGION = "us-east-1"
CW_NAMESPACE  = "KrishiRakshak"
CW_REGION     = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

_bedrock = None
_cw      = None


def _get_bedrock():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    return _bedrock


def _get_cw():
    global _cw
    if _cw is None:
        _cw = boto3.client("cloudwatch", region_name=CW_REGION)
    return _cw


# ── Judge ─────────────────────────────────────────────────────────────────────

def judge_rag(query: str, context_chunks: List[Dict], answer: str) -> Dict[str, float]:
    """
    Use Llama 3.1 8B to score faithfulness and relevance.
    Returns scores between 0.0 and 1.0.
    """
    context_text = "\n---\n".join(
        c.get("chunk", c.get("text", "")) for c in context_chunks[:3]
    )[:2000]  # cap context to keep costs low

    prompt = f"""You are an evaluation judge for a crop disease RAG system.
Score the response on four dimensions. Reply with ONLY a JSON object, nothing else.

Query: {query}

Retrieved Context:
{context_text}

Answer: {answer}

Scoring rules:
- faithfulness (0.0 to 1.0): Is the answer grounded in the context without contradiction?
- answer_relevance (0.0 to 1.0): Does the answer directly address what the user asked?
- context_relevance (0.0 to 1.0): How relevant is the retrieved context to the query? (retriever quality)
- context_precision (0.0 to 1.0): Is the context concise and free of noise? Low = too much irrelevant content retrieved.

Reply with exactly this JSON:
{{"faithfulness": <float>, "answer_relevance": <float>, "context_relevance": <float>, "context_precision": <float>}}"""

    try:
        resp = _get_bedrock().converse(
            modelId=JUDGE_MODEL,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 80, "temperature": 0.0},
        )
        raw = resp["output"]["message"]["content"][0]["text"].strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        scores = json.loads(raw[start:end])
        return {
            "faithfulness"     : float(max(0.0, min(1.0, scores.get("faithfulness", 0.5)))),
            "answer_relevance" : float(max(0.0, min(1.0, scores.get("answer_relevance", 0.5)))),
            "context_relevance": float(max(0.0, min(1.0, scores.get("context_relevance", 0.5)))),
            "context_precision": float(max(0.0, min(1.0, scores.get("context_precision", 0.5)))),
        }
    except Exception as e:
        logger.warning(f"RAG judge failed: {e}")
        return {"faithfulness": -1.0, "answer_relevance": -1.0, "context_relevance": -1.0, "context_precision": -1.0}


# ── CloudWatch publisher ───────────────────────────────────────────────────────

def _push_metric(name: str, value: float, unit: str = "None", dimensions: list = None):
    """Push a single metric to CloudWatch."""
    try:
        metric = {
            "MetricName": name,
            "Value"     : value,
            "Unit"      : unit,
            "Dimensions": dimensions or [],
        }
        _get_cw().put_metric_data(Namespace=CW_NAMESPACE, MetricData=[metric])
    except Exception as e:
        logger.warning(f"CloudWatch push failed for {name}: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

def log_rag_request(
    query         : str,
    context_chunks: List[Dict],
    answer        : str,
    latency_ms    : float,
    guardrail_blocked: bool = False,
):
    """
    Score RAG response and push metrics to CloudWatch.
    Call this after every /v1/query response.
    Non-blocking — errors are swallowed so monitoring never breaks the API.
    """
    try:
        # Guardrail block rate
        _push_metric("GuardrailBlocked", 1.0 if guardrail_blocked else 0.0)
        _push_metric("LatencyMs", latency_ms, unit="Milliseconds")

        if guardrail_blocked or not context_chunks:
            return  # no RAG to score

        scores = judge_rag(query, context_chunks, answer)
        if scores["faithfulness"] >= 0:
            _push_metric("Faithfulness",      scores["faithfulness"])
            _push_metric("AnswerRelevance",   scores["answer_relevance"])
            _push_metric("ContextRelevance",  scores["context_relevance"])
            _push_metric("ContextPrecision",  scores["context_precision"])
            logger.info(
                f"RAG metrics — faithfulness={scores['faithfulness']:.2f} "
                f"answer_relevance={scores['answer_relevance']:.2f} "
                f"context_relevance={scores['context_relevance']:.2f} "
                f"context_precision={scores['context_precision']:.2f} "
                f"latency={latency_ms:.0f}ms"
            )
    except Exception as e:
        logger.warning(f"log_rag_request failed: {e}")


def log_classifier_request(disease: str, confidence: float, latency_ms: float):
    """
    Push classifier metrics to CloudWatch.
    Call this after every /v1/diagnose response.
    """
    try:
        _push_metric(
            "ClassifierConfidence",
            confidence,
            dimensions=[{"Name": "Disease", "Value": disease}],
        )
        _push_metric("LatencyMs", latency_ms, unit="Milliseconds")
        logger.info(f"Classifier metrics — disease={disease} confidence={confidence:.1f}%")
    except Exception as e:
        logger.warning(f"log_classifier_request failed: {e}")
