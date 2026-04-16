"""
chunker.py
----------
Language-aware text chunker for Hindi and English.

- English : RecursiveCharacterTextSplitter (fixed character length)
- Hindi   : Split on । (danda) with min char accumulation + 20% overlap
- Language detection : Lingua (same detector used in agent)
- Entity extraction  : keyword mapping for disease/crop/pesticide metadata
"""

import re
import logging
from datetime import date
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from lingua import Language, LanguageDetectorBuilder

logger = logging.getLogger(__name__)

# ── Language detector ─────────────────────────────────────────────────────────
_detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.HINDI,
).build()


def detect_language(text: str) -> str:
    lang = _detector.detect_language_of(text)
    return "hi" if lang == Language.HINDI else "en"


# ── Entity keyword maps ───────────────────────────────────────────────────────
DISEASE_LIST = [
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Potato Late Blight",
    "Tomato Leaf Mold",
    "Corn Common Rust",
]

CROP_LIST = ["Tomato", "Potato", "Corn"]

PESTICIDE_LIST = [
    "Mancozeb", "Ridomil Gold", "Metalaxyl", "Propiconazole",
    "Chlorothalonil", "Indofil M-45", "Tilt 25 EC", "Kavach",
    "Blitox", "Curzate", "Sectin", "Folicur",
]

HINDI_MAP: Dict[str, tuple] = {
    "टमाटर"         : ("crop",     "Tomato"),
    "आलू"           : ("crop",     "Potato"),
    "मक्का"         : ("crop",     "Corn"),
    "अर्ली ब्लाइट"  : ("disease",  "Tomato Early Blight"),
    "लेट ब्लाइट"    : ("disease",  "Tomato Late Blight"),
    "लीफ मोल्ड"     : ("disease",  "Tomato Leaf Mold"),
    "कॉमन रस्ट"     : ("disease",  "Corn Common Rust"),
    "मैन्कोज़ेब"     : ("pesticide","Mancozeb"),
    "प्रोपिकोनाज़ोल" : ("pesticide","Propiconazole"),
    "रिडोमिल"       : ("pesticide","Ridomil Gold"),
}


def extract_entities(text: str) -> Dict[str, List[str]]:
    text_lower = text.lower()
    diseases   = [d for d in DISEASE_LIST   if d.lower() in text_lower]
    crops      = [c for c in CROP_LIST      if c.lower() in text_lower]
    pesticides = [p for p in PESTICIDE_LIST if p.lower() in text_lower]

    for hindi_word, (entity_type, english_name) in HINDI_MAP.items():
        if hindi_word in text:
            if entity_type == "crop"      and english_name not in crops:      crops.append(english_name)
            if entity_type == "disease"   and english_name not in diseases:   diseases.append(english_name)
            if entity_type == "pesticide" and english_name not in pesticides: pesticides.append(english_name)

    return {
        "disease_mentioned"  : diseases,
        "crop_mentioned"     : crops,
        "pesticide_mentioned": pesticides,
    }


def chunk_text(
    text      : str,
    source    : str = "unknown",
    en_chunk_size : int = 400,
    en_overlap    : int = 50,
    hi_min_chars  : int = 200,
) -> List[Dict[str, Any]]:
    """
    Split text into chunks with metadata.

    Each chunk dict contains:
        text, source, chunk_index, language, ingested_at,
        disease_mentioned, crop_mentioned, pesticide_mentioned
    """
    sentences = re.split(r'(?<=[।.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    # Detect language once on the first 500 chars — avoids calling Lingua
    # for every sentence (hundreds of calls on large documents).
    doc_lang = detect_language(text[:500])
    tagged   = [(s, doc_lang) for s in sentences]

    # Group consecutive same-language sentences
    groups: List[tuple] = []
    cur_lang, cur_group = tagged[0][1], []
    for sent, lang in tagged:
        if lang == cur_lang:
            cur_group.append(sent)
        else:
            groups.append((cur_lang, cur_group))
            cur_lang, cur_group = lang, [sent]
    groups.append((cur_lang, cur_group))

    chunks      : List[Dict[str, Any]] = []
    chunk_index : int = 0

    # Create splitter once — not inside the loop
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=en_chunk_size, chunk_overlap=en_overlap
    )

    for lang, sents in groups:
        if lang == "hi":
            current = ""
            for sent in sents:
                current += sent + " "
                if len(current) >= hi_min_chars:
                    entities = extract_entities(current)
                    chunks.append({
                        "text"       : current.strip(),
                        "source"     : source,
                        "chunk_index": chunk_index,
                        "language"   : "hi",
                        "ingested_at": str(date.today()),
                        **entities,
                    })
                    chunk_index  += 1
                    overlap_chars = int(len(current) * 0.2)
                    current       = current[-overlap_chars:].strip() + " "

            if current.strip():
                entities = extract_entities(current)
                if chunks and len(current.strip()) < hi_min_chars // 2:
                    chunks[-1]["text"] += " " + current.strip()
                else:
                    chunks.append({
                        "text"       : current.strip(),
                        "source"     : source,
                        "chunk_index": chunk_index,
                        "language"   : "hi",
                        "ingested_at": str(date.today()),
                        **entities,
                    })
                    chunk_index += 1

        else:
            for chunk in splitter.split_text(" ".join(sents)):
                entities = extract_entities(chunk)
                chunks.append({
                    "text"       : chunk,
                    "source"     : source,
                    "chunk_index": chunk_index,
                    "language"   : "en",
                    "ingested_at": str(date.today()),
                    **entities,
                })
                chunk_index += 1

    logger.info(f"Chunked '{source}' → {len(chunks)} chunks")
    return chunks
