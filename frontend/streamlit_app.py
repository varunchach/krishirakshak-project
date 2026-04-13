"""KrishiRakshak — Streamlit Demo UI."""

import requests
import streamlit as st

st.set_page_config(page_title="KrishiRakshak", page_icon="🌾", layout="wide")

API_URL = "http://localhost:8000/v1"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌾 KrishiRakshak")
st.caption("AI-Powered Crop Disease Diagnosis — Hindi & English")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    session_id = st.text_input("Session ID", value="demo-session")
    st.divider()
    st.markdown("**LLM:** Claude Sonnet 4.6 (Bedrock)")
    st.markdown("**Embeddings:** BGE-M3 (SageMaker)")
    st.markdown("**Classifier:** EfficientNet-B3")
    st.markdown("**Audio:** Amazon Polly")
    st.divider()
    st.markdown("**Supported crops:** Tomato · Potato · Corn")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🖼️ Diagnose Image", "💬 Ask a Question", "📄 Upload PDF"])

# ── Tab 1: Image Diagnosis ────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload a leaf image to diagnose disease")
    uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded, caption="Uploaded Image", use_column_width=True)

        with col2:
            if st.button("🔍 Diagnose", key="diagnose"):
                with st.spinner("Analysing..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/diagnose",
                            files={"image": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                            data={"session_id": session_id},
                            timeout=60,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.success("Diagnosis complete")
                            st.markdown(f"**Answer:**\n\n{data['answer']}")
                            st.caption(f"Language: {data['language']} | Latency: {data['latency_ms']} ms")
                            if data.get("audio_url"):
                                st.audio(data["audio_url"])
                        else:
                            st.error(f"Error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API. Is the server running on localhost:8000?")

# ── Tab 2: Text Query ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("Ask in Hindi or English")
    query = st.text_area(
        "Your question",
        placeholder="e.g. टमाटर की पत्तियों पर भूरे धब्बे हो रहे हैं, क्या करूं?\nor: How to treat tomato early blight?",
        height=100,
    )

    if st.button("🤖 Ask Agent", key="query") and query.strip():
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"query": query, "session_id": session_id},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown(f"**Answer:**\n\n{data['answer']}")
                    st.caption(f"Language: {data['language']} | Latency: {data['latency_ms']} ms")
                    if data.get("audio_url"):
                        st.audio(data["audio_url"])
                else:
                    st.error(f"Error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is the server running on localhost:8000?")

# ── Tab 3: PDF Ingest ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Upload a government / ICAR PDF to add to knowledge base")
    pdf = st.file_uploader("Choose PDF", type=["pdf"])
    source_name = st.text_input("Document name (optional)", placeholder="e.g. ICAR Tomato Guide 2023")

    if st.button("📥 Ingest PDF", key="ingest") and pdf:
        with st.spinner("Processing..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ingest",
                    files={"pdf": (pdf.name, pdf.getvalue(), "application/pdf")},
                    data={"source": source_name or pdf.name},
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Added {data['chunks_added']} chunks from '{data['source']}'")
                else:
                    st.error(f"Error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is the server running on localhost:8000?")
