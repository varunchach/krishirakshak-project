"""
rag_generator.py
----------------
RAG generator using Claude Sonnet via Amazon Bedrock.

Rules:
- Respond in SAME language as query (Hindi → Hindi, English → English)
- Plain prose only — no markdown, bullets, asterisks (TTS-safe)
- Response capped at ~150 words
- Must cite at least one India-specific pesticide with dosage
- Never fabricate disease names, dosages, or pesticide names
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional

import boto3

logger = logging.getLogger(__name__)

BEDROCK_REGION    = "us-east-1"
INFERENCE_PROFILE = "us.anthropic.claude-sonnet-4-6-20251001-v1:0"

SYSTEM_PROMPT = """You are KrishiRakshak, an agricultural expert helping Indian farmers diagnose and treat crop diseases.

STRICT RULES — follow every rule exactly:
1. Respond in the SAME language the farmer used. Hindi query → Hindi answer. English query → English answer. Never switch languages.
2. Write in plain prose only. No bullet points, no numbered lists, no markdown, no asterisks, no headers, no emojis. Plain sentences only — this output will be read aloud.
3. Keep the response under 150 words.
4. Mention at least one pesticide brand available in India with dosage.
5. End with one short prevention tip for next season.
6. If the context does not contain enough information, say so in the farmer's language and advise them to contact their local Krishi Vigyan Kendra.
7. Never guess or fabricate disease names, dosages, or pesticide names. Use only what is in the context."""


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("chunk") or chunk.get("text", "")
        if text.strip():
            lines.append(f"[{i}] {text.strip()}")
    return "\n\n".join(lines) if lines else "No relevant context found."


class RAGGenerator:
    def __init__(
        self,
        region: str = BEDROCK_REGION,
        model_id: str = INFERENCE_PROFILE,
        max_tokens: int = 300,
        temperature: float = 0.1,
    ):
        self.model_id    = model_id
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.client      = boto3.client("bedrock-runtime", region_name=region)
        logger.info(f"RAGGenerator ready (model={model_id})")

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate treatment advisory from retrieved chunks.

        Args:
            query  : user query in any language
            chunks : retrieved chunk dicts (must have 'chunk' or 'text' key)

        Returns:
            {answer, latency_ms, model, chunks_used}
        """
        context      = _build_context(chunks)
        user_message = (
            f"Question: {query}\n\n"
            f"Relevant context:\n{context}\n\n"
            f"Answer in the same language as the question. Plain prose only."
        )

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens"       : max_tokens or self.max_tokens,
            "temperature"      : self.temperature,
            "system"           : SYSTEM_PROMPT,
            "messages"         : [{"role": "user", "content": user_message}],
        }

        start = time.monotonic()
        try:
            resp    = self.client.invoke_model(modelId=self.model_id, body=json.dumps(body))
            result  = json.loads(resp["Body"].read())
            answer  = result["content"][0]["text"].strip()
            latency = (time.monotonic() - start) * 1000
            logger.info(f"Generated {len(answer)} chars in {latency:.0f}ms")
            return {
                "answer"      : answer,
                "latency_ms"  : round(latency, 1),
                "model"       : self.model_id,
                "chunks_used" : len(chunks),
            }
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            logger.error(f"RAGGenerator failed: {e}")
            return {
                "answer"      : "Unable to generate advice. Please consult your local Krishi Vigyan Kendra.",
                "latency_ms"  : round(latency, 1),
                "model"       : self.model_id,
                "chunks_used" : len(chunks),
                "error"       : str(e),
            }


# Module-level singleton — initialized on first import
_generator: Optional[RAGGenerator] = None


def get_generator() -> RAGGenerator:
    global _generator
    if _generator is None:
        _generator = RAGGenerator()
    return _generator


def generate(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Convenience function — returns answer string directly."""
    return get_generator().generate(query, chunks)["answer"]
