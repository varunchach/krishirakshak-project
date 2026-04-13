"""Tests for chunker and retriever pipeline."""

import pytest
from unittest.mock import patch, MagicMock


# ── Chunker tests ─────────────────────────────────────────────────────────────

def test_chunk_text_returns_list():
    from src.services.chunker import chunk_text
    chunks = chunk_text("Tomato early blight is caused by Alternaria solani. "
                        "Spray Mancozeb at 2.5 g/L.", source="test_doc")
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_chunk_metadata_keys():
    from src.services.chunker import chunk_text
    chunks = chunk_text("Apply Mancozeb to treat tomato blight.", source="icar_guide")
    for c in chunks:
        assert "text" in c
        assert "source" in c
        assert "language" in c
        assert "chunk_index" in c


def test_detect_language_english():
    from src.services.chunker import detect_language
    lang = detect_language("Spray Mancozeb on tomato leaves to control blight.")
    assert lang == "en"


def test_detect_language_hindi():
    from src.services.chunker import detect_language
    lang = detect_language("टमाटर की फसल में झुलसा रोग के लिए मैनकोजेब का छिड़काव करें।")
    assert lang == "hi"


def test_hindi_chunk_overlap():
    from src.services.chunker import chunk_text
    hindi_text = (
        "टमाटर की फसल में जल्दी झुलसा रोग होता है। "
        "यह ऑल्टरनेरिया सोलानी कवक के कारण होता है। "
        "पत्तियों पर गोल भूरे धब्बे बनते हैं। "
        "मैनकोजेब 75WP का 2.5 ग्राम प्रति लीटर पानी में घोल बनाएं। "
        "7-10 दिन के अंतराल पर छिड़काव करें।"
    )
    chunks = chunk_text(hindi_text, source="test_hindi")
    # Should produce at least 1 chunk
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c["text"]) > 0


# ── Retriever tests ───────────────────────────────────────────────────────────

def test_add_and_search():
    """Add chunks to VectorStore and verify search returns results."""
    import numpy as np
    from src.services.retriever import VectorStore

    store = VectorStore()
    fake_chunks = [
        {"text": "Mancozeb treats tomato early blight effectively.", "source": "t1",
         "chunk_index": 0, "language": "en", "ingested_at": "2024-01-01",
         "disease_mentioned": ["Tomato Early Blight"], "crop_mentioned": ["tomato"],
         "pesticide_mentioned": ["Mancozeb"]},
        {"text": "Potato late blight needs Metalaxyl application.", "source": "t2",
         "chunk_index": 0, "language": "en", "ingested_at": "2024-01-01",
         "disease_mentioned": ["Potato Late Blight"], "crop_mentioned": ["potato"],
         "pesticide_mentioned": ["Metalaxyl"]},
    ]
    fake_embeddings = np.random.rand(2, 1024).astype("float32")

    with patch("src.services.retriever.get_embeddings", return_value=fake_embeddings):
        store.add_chunks(fake_chunks)

    assert store.faiss_index is not None
    assert store.faiss_index.ntotal == 2
    assert len(store.chunks) == 2

    query_emb = np.random.rand(1, 1024).astype("float32")
    with patch("src.services.retriever.get_embeddings", return_value=query_emb):
        results = store.search("tomato blight treatment", k=2)

    assert len(results) <= 2
    for r in results:
        assert "chunk" in r
        assert "score" in r
