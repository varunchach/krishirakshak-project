"""
tools.py
--------
LangChain @tool definitions for the KrishiRakshak LangGraph agent.

Each tool wraps a core service function.
Raw service functions are importable separately for direct use.
"""

import json
import logging
import urllib.parse
import urllib.request
import re

from langchain_core.tools import tool

from src.models.classifier  import predict as _predict
from src.models.rag_generator import generate as _generate
from src.services.retriever import get_store
from src.services.audio     import text_to_speech
from src.services.chunker   import detect_language

logger = logging.getLogger(__name__)


@tool
def image_diagnosis_tool(image_path: str) -> str:
    """
    Diagnoses crop disease from a leaf image.
    Use when user uploads an image (jpg/png).
    Returns disease name, confidence score.
    """
    from src.models.classifier import load_model
    import os
    model_path = os.getenv("CLASSIFIER_MODEL_PATH", "models_pkl/best_model.pth")

    # Lazy load — cache on module after first call
    if not hasattr(image_diagnosis_tool, "_model"):
        image_diagnosis_tool._model = load_model(model_path)

    result = _predict(image_diagnosis_tool._model, image_path)
    if result["low_conf"]:
        return f"Low confidence ({result['confidence']}%). Please upload a clearer image or consult your local Krishi Vigyan Kendra."
    return json.dumps(result)


@tool
def retriever_tool(query: str) -> str:
    """
    Retrieves relevant crop disease information from the knowledge base.
    Use for any text query about diseases, treatments, or pesticides.
    """
    store   = get_store()
    results = store.search(query, k=5)
    return json.dumps(results)


@tool
def web_search_tool(query: str) -> str:
    """
    Searches ICAR and government agriculture sources.
    Use when retriever returns no relevant results or user asks about latest policies.
    """
    try:
        q    = urllib.parse.quote(query + " crop disease site:icar.org.in OR site:agricoop.nic.in")
        url  = f"https://html.duckduckgo.com/html/?q={q}"
        req  = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")
        snippets = re.findall(r'class="result__snippet">(.*?)</a>', html)
        return " ".join(snippets[:3]) if snippets else "No relevant results found."
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return "Web search unavailable."


@tool
def direct_context_tool(query: str, raw_text: str) -> str:
    """
    Answers query using full document text directly.
    Use only when uploaded PDF is small (fits in context window).
    """
    import boto3, json as _json
    from src.models.rag_generator import INFERENCE_PROFILE, BEDROCK_REGION, SYSTEM_PROMPT
    bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    resp    = bedrock.invoke_model(
        modelId=INFERENCE_PROFILE,
        body=_json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "temperature": 0.1,
            "system"  : SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": f"Question: {query}\n\nDocument:\n{raw_text}"}],
        }),
    )
    return _json.loads(resp["Body"].read())["content"][0]["text"].strip()


@tool
def rag_generator_tool(query: str, chunks_json: str) -> str:
    """
    Generates final treatment advice from retrieved context chunks.
    Responds in the same language as the query.
    Use after retriever_tool or web_search_tool.
    """
    chunks = json.loads(chunks_json)
    return _generate(query, chunks)


@tool
def audio_generation_tool(text: str, language: str) -> str:
    """
    Converts final text response to MP3 audio using Amazon Polly.
    Always call this as the last step to deliver audio to the farmer.
    Language must be 'en' or 'hi'.
    """
    lang_code = "hi" if language.lower() in ["hi", "hindi", "हिंदी"] else "en"
    audio_bytes = text_to_speech(text, lang_code)
    # In production, upload to S3 and return presigned URL
    # For now return confirmation with byte size
    return json.dumps({
        "status"    : "success",
        "language"  : lang_code,
        "size_bytes": len(audio_bytes),
    })


# ── Tool registry for LangGraph ───────────────────────────────────────────────
# rag_generator_tool removed — agent (Claude) writes the final answer itself.
# Keeping it caused an extra Claude call inside an already-running Claude call.
ALL_TOOLS = [
    image_diagnosis_tool,
    retriever_tool,
    web_search_tool,
    direct_context_tool,
]
