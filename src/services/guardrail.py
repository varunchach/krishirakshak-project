"""
guardrail.py
------------
Input safety classification using Llama 3.1 8B on Bedrock.
Blocks off-topic and harmful queries before they reach the RAG pipeline.
"""

import logging
import boto3

logger = logging.getLogger(__name__)

MODEL_ID = "us.meta.llama3-1-8b-instruct-v1:0"
REGION   = "us-east-1"

CLASSIFICATION_PROMPT = """You are a safety classifier for KrishiRakshak, an AI assistant that ONLY handles:
- Crop diseases (Tomato, Potato, Corn)
- Disease treatments and pesticides
- Agricultural farming advice
- Government farming schemes and policies

Classify the user message as SAFE or UNSAFE.
SAFE: questions about crops, diseases, treatments, pesticides, farming, agriculture, seeds, soil, irrigation, government farm schemes.
UNSAFE: anything unrelated to farming or agriculture — coding, politics, general knowledge, entertainment, harmful content, personal advice.

Respond with exactly one word: SAFE or UNSAFE.

User message: {query}
Classification:"""

BLOCKED_RESPONSE = (
    "I can only assist with crop diseases and farming advice for Tomato, Potato, and Corn. "
    "Please ask me about disease symptoms, treatments, or agricultural practices."
)


def check(query: str) -> tuple[bool, str]:
    """
    Classify query as safe or unsafe.

    Returns:
        (is_safe: bool, reason: str)
    """
    try:
        client = boto3.client("bedrock-runtime", region_name=REGION)
        prompt = CLASSIFICATION_PROMPT.format(query=query[:500])  # cap input length

        resp = client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 5, "temperature": 0.0},
        )
        verdict = resp["output"]["message"]["content"][0]["text"].strip().upper()
        is_safe = verdict.startswith("SAFE")
        logger.info(f"Guardrail: '{query[:60]}...' → {verdict}")
        return is_safe, verdict

    except Exception as e:
        # Fail open — if guardrail errors, allow the request through
        logger.warning(f"Guardrail check failed ({e}) — allowing request")
        return True, "GUARDRAIL_ERROR"
