"""KrishiRakshak — Chat UI"""

import uuid
import requests
import streamlit as st

st.set_page_config(
    page_title="KrishiRakshak",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000/v1"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu, footer { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #052e16 0%, #14532d 100%);
    padding-top: 0;
}
[data-testid="stSidebar"] * { color: #d1fae5 !important; }
[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: #d1fae5 !important;
    border-radius: 8px;
    font-size: 0.85rem;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.15);
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 1px dashed rgba(255,255,255,0.2);
    border-radius: 10px;
    padding: 8px;
}

/* ── Main area ── */
.block-container {
    padding: 0 2.5rem 2rem 2.5rem;
    max-width: 860px;
    margin: 0 auto;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #052e16 0%, #166534 60%, #15803d 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    box-shadow: 0 4px 24px rgba(5,46,22,0.18);
}
.hero-icon { font-size: 3.2rem; line-height: 1; }
.hero-text h1 {
    font-size: 1.9rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.hero-text p {
    font-size: 0.92rem;
    color: #86efac;
    margin: 0;
    font-weight: 400;
}
.hero-pills {
    display: flex;
    gap: 8px;
    margin-top: 10px;
    flex-wrap: wrap;
}
.pill {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.73rem;
    color: #d1fae5 !important;
    font-weight: 500;
}

/* ── Chat messages — left/right layout ── */
[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    margin-bottom: 0.6rem;
    padding: 0.6rem 1rem;
    max-width: 75%;
}
/* Bot messages — left aligned */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    align-self: flex-start;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    margin-right: auto;
}
/* User messages — right aligned */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    align-self: flex-end;
    background: #052e16;
    border: 1px solid #166534;
    margin-left: auto;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) * {
    color: #d1fae5 !important;
}
/* Make the messages container flex column */
[data-testid="stChatMessageContainer"] {
    display: flex;
    flex-direction: column;
}
[data-testid="stChatMessageContent"] p {
    font-size: 0.93rem;
    line-height: 1.65;
}

/* ── Input bar ── */
[data-testid="stChatInput"] {
    border-radius: 12px !important;
    border: 1.5px solid #d1fae5 !important;
    font-size: 0.92rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #16a34a !important;
    box-shadow: 0 0 0 3px rgba(22,163,74,0.12) !important;
}

/* ── Upload strip ── */
.upload-strip {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    margin-bottom: 8px;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #16a34a !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.04) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.5rem 0.5rem 1rem 0.5rem; text-align: center;'>
        <div style='font-size: 2.8rem;'>🌾</div>
        <div style='font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-top: 6px;'>KrishiRakshak</div>
        <div style='font-size: 0.75rem; color: #86efac; margin-top: 2px;'>AI for Indian Farmers</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    generate_audio = st.toggle("Voice Response", value=False)

    st.divider()

    st.markdown("<div style='font-size:0.78rem; font-weight:600; color:#86efac; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>Knowledge Base</div>", unsafe_allow_html=True)

    with st.expander("Upload PDF Document"):
        pdf = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
        if pdf and st.button("Ingest Document", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    r = requests.post(
                        f"{API_URL}/ingest",
                        files={"pdf": (pdf.name, pdf.getvalue(), "application/pdf")},
                        data={"source": pdf.name},
                        timeout=120,
                    )
                    if r.status_code == 200:
                        st.success(f"{r.json()['chunks_added']} chunks added")
                    else:
                        st.error(f"Error {r.status_code}")
                except Exception as e:
                    st.error(str(e))

    with st.expander("Diagnose Crop Disease"):
        leaf_image = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if leaf_image and st.button("Analyse Leaf", use_container_width=True):
            if st.session_state.get("last_image") != leaf_image.name:
                st.session_state.last_image = leaf_image.name
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"Diagnose this leaf: **{leaf_image.name}**",
                    "image": leaf_image.getvalue(),
                })
                with st.spinner("Analysing..."):
                    try:
                        r = requests.post(
                            f"{API_URL}/diagnose",
                            files={"image": (leaf_image.name, leaf_image.getvalue(), leaf_image.type)},
                            data={"session_id": st.session_state.session_id, "generate_audio": str(generate_audio).lower()},
                            timeout=90,
                        )
                        if r.status_code == 200:
                            d = r.json()
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": d["answer"],
                                "audio_url": d.get("audio_url"),
                            })
                        else:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Error: {r.json().get('detail', r.text)}",
                            })
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": f"Cannot reach API: {e}"})
                st.rerun()

    st.divider()

    st.markdown("<div style='font-size:0.78rem; font-weight:600; color:#86efac; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>Controls</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    st.markdown(f"<div style='font-size:0.7rem; color:#6b7280; text-align:center;'>Session: {st.session_state.session_id}</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.7rem; color:#4ade80; text-align:center; margin-top:8px; line-height:1.6;'>
        Tomato · Potato · Corn<br>
        Hindi · English
    </div>
    """, unsafe_allow_html=True)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">🌱</div>
    <div class="hero-text">
        <h1>KrishiRakshak</h1>
        <p>Detect crop diseases, get treatment advice, and query farming documents — in Hindi or English.</p>
        <div class="hero-pills">
            <span class="pill">Crop Disease Detection</span>
            <span class="pill">Document Q&A</span>
            <span class="pill">Hindi · English</span>
            <span class="pill">Voice Responses</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image"):
            st.image(msg["image"], width=240)
        st.markdown(msg["content"])
        if msg.get("audio_url"):
            st.audio(msg["audio_url"])

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about crop diseases, treatments, or farming policies...")

# ── Handle text query ─────────────────────────────────────────────────────────
if user_input and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        try:
            r = requests.post(
                f"{API_URL}/query",
                json={
                    "query": user_input,
                    "session_id": st.session_state.session_id,
                    "generate_audio": generate_audio,
                },
                timeout=90,
            )
            if r.status_code == 200:
                d = r.json()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": d["answer"],
                    "audio_url": d.get("audio_url"),
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {r.json().get('detail', r.text)}",
                })
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Cannot reach API: {e}"})
    st.rerun()
