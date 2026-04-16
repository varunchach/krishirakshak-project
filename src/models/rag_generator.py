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
INFERENCE_PROFILE = "us.anthropic.claude-sonnet-4-6"

SYSTEM_PROMPT = """You are KrishiRakshak, an agricultural expert helping Indian farmers diagnose and treat crop diseases.

STRICT RULES — follow every rule exactly:
1. Respond in the SAME language the farmer used. Hindi query → Hindi answer. English query → English answer. Never switch languages.
2. Write in plain prose only. No bullet points, no numbered lists, no markdown, no asterisks, no headers, no emojis. Plain sentences only — this output will be read aloud.
3. Keep the response under 150 words.
4. Mention at least one pesticide brand available in India with dosage.
5. End with one short prevention tip for next season.
6. If the context does not contain enough information, say so in the farmer's language and advise them to contact their local Krishi Vigyan Kendra.
7. Never guess or fabricate disease names, dosages, or pesticide names. Use only what is in the context.
8. When answering from retrieved context, mention the source document name at the end in parentheses, e.g. "(Source: filename.pdf)". If multiple sources contributed, list all of them."""


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        text   = chunk.get("chunk") or chunk.get("text", "")
        source = chunk.get("metadata", {}).get("source", "") or chunk.get("source", "")
        if text.strip():
            header = f"[{i}]" + (f" (Source: {source})" if source else "")
            lines.append(f"{header}\n{text.strip()}")
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

        start = time.monotonic()
        try:
            resp   = self.client.converse(
                modelId=self.model_id,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": user_message}]}],
                inferenceConfig={"maxTokens": max_tokens or self.max_tokens, "temperature": self.temperature},
            )
            answer = resp["output"]["message"]["content"][0]["text"].strip()
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

# LRU cache capped at 256 entries — prevents unbounded memory growth in
# long-running production containers.
_cache: dict = {}
_CACHE_MAX  = 256


def get_generator() -> RAGGenerator:
    global _generator
    if _generator is None:
        _generator = RAGGenerator()
    return _generator


def generate(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Convenience function — returns answer string directly."""
    key = query.strip().lower()
    if key in _cache:
        logger.info(f"Cache hit for query: {key[:60]}")
        return _cache[key]
    answer = get_generator().generate(query, chunks)["answer"]
    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))  # evict oldest entry
    _cache[key] = answer
    return answer


def generate_direct(
    query    : str,
    raw_text : str,
    history  : Optional[List[tuple]] = None,
) -> str:
    """
    Answer query using full document text — no retrieval needed.
    Used for small documents that fit within Claude's context window.

    Args:
        query    : current user question
        raw_text : full document text (capped at 60k chars)
        history  : list of (user_query, assistant_answer) tuples — last N exchanges
    """
    gen = get_generator()

    # Build conversation turns from history so Claude has session context
    messages = []
    if history:
        for past_query, past_answer in history:
            messages.append({"role": "user",      "content": [{"text": past_query}]})
            messages.append({"role": "assistant",  "content": [{"text": past_answer}]})
        logger.info(f"generate_direct: injecting {len(history)} prior exchange(s) into context")

    # Current turn — document context included only in the current user message
    user_message = (
        f"Question: {query}\n\n"
        f"Full document context:\n{raw_text[:60000]}\n\n"
        f"Answer in the same language as the question. Plain prose only."
    )
    messages.append({"role": "user", "content": [{"text": user_message}]})

    try:
        resp   = gen.client.converse(
            modelId=gen.model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            inferenceConfig={"maxTokens": 300, "temperature": 0.1},
        )
        answer = resp["output"]["message"]["content"][0]["text"].strip()
        logger.info(f"generate_direct: {len(answer)} chars (direct context, no retrieval)")
        return answer
    except Exception as e:
        logger.error(f"generate_direct failed: {e}")
        return "Unable to generate advice. Please consult your local Krishi Vigyan Kendra."
