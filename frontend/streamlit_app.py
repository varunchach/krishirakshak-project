"""KrishiRakshak — Streamlit Demo UI."""

import base64

import requests
import streamlit as st

st.set_page_config(page_title="KrishiRakshak", page_icon="\U0001F33E", layout="wide")

# Config
API_URL = "http://localhost:8000/v1"

LANGUAGES = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Marathi": "mr-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Kannada": "kn-IN",
    "Bengali": "bn-IN",
}

SEVERITY_COLORS = {
    "none": "\U0001F7E2",
    "moderate": "\U0001F7E1",
    "high": "\U0001F7E0",
    "critical": "\U0001F534",
}

# Header
st.title("\U0001F33E KrishiRakshak")
st.caption("AI-Powered Crop Disease Diagnosis with Voice Output in Indian Languages")
st.divider()

# Sidebar
with st.sidebar:
    st.header("Settings")
    lang_name = st.selectbox("Output Language", list(LANGUAGES.keys()))
    lang_code = LANGUAGES[lang_name]
    include_audio = st.checkbox("Include Voice Output", value=True)
    st.divider()
    st.markdown("**Model:** Florence-2 (fine-tuned)")
    st.markdown("**Translation:** Sarvam Mayura")
    st.markdown("**TTS:** Sarvam Bulbul")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Leaf Image")
    uploaded = st.file_uploader(
        "Take a photo or upload an image of a diseased leaf",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded:
        st.image(uploaded, caption="Uploaded Leaf", use_container_width=True)

with col2:
    st.subheader("Diagnosis Results")

    if uploaded and st.button("\U0001F50D Diagnose", type="primary", use_container_width=True):
        with st.spinner("Analyzing leaf image..."):
            try:
                files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                data = {"language": lang_code, "include_audio": str(include_audio).lower()}

                resp = requests.post(f"{API_URL}/diagnose", files=files, data=data, timeout=60)

                if resp.status_code != 200:
                    st.error(f"Error: {resp.json().get('detail', {}).get('message', 'Unknown error')}")
                else:
                    result = resp.json()
                    disease = result["disease"]
                    treatment = result["treatment"]

                    # Disease info
                    severity_icon = SEVERITY_COLORS.get(disease.get("severity", ""), "\u26AA")
                    st.metric("Disease Detected", disease["name"])

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Confidence", f"{disease['confidence']:.0%}")
                    m2.metric("Severity", f"{severity_icon} {disease.get('severity', 'N/A').title()}")
                    m3.metric("Crop", disease.get("crop", "N/A"))

                    # Treatment (English)
                    st.subheader("Treatment Advice (English)")
                    st.write(treatment["english"])

                    # Translated treatment
                    if lang_code != "en-IN" and treatment.get("translated"):
                        st.subheader(f"Treatment Advice ({lang_name})")
                        st.write(treatment["translated"])

                    # Audio playback
                    audio = result.get("audio")
                    if audio and audio.get("base64"):
                        st.subheader(f"\U0001F50A Listen in {lang_name}")
                        audio_bytes = base64.b64decode(audio["base64"])
                        st.audio(audio_bytes, format="audio/wav")

                    # Metadata
                    with st.expander("Technical Details"):
                        meta = result.get("metadata", {})
                        st.json({
                            "request_id": meta.get("request_id"),
                            "model_version": meta.get("model_version"),
                            "inference_time_ms": meta.get("inference_time_ms"),
                            "total_time_ms": meta.get("total_time_ms"),
                        })

                    # Feedback
                    st.divider()
                    st.write("**Was this diagnosis correct?**")
                    fc1, fc2 = st.columns(2)
                    if fc1.button("\U0001F44D Correct", use_container_width=True):
                        requests.post(f"{API_URL}/feedback", json={
                            "request_id": result["request_id"],
                            "is_correct": True,
                        })
                        st.success("Thanks! Feedback recorded.")
                    if fc2.button("\U0001F44E Incorrect", use_container_width=True):
                        requests.post(f"{API_URL}/feedback", json={
                            "request_id": result["request_id"],
                            "is_correct": False,
                        })
                        st.warning("Thanks! We'll use this to improve.")

            except requests.ConnectionError:
                st.error("Cannot connect to API. Ensure the backend is running: `uvicorn src.api.main:app`")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    elif not uploaded:
        st.info("Upload a leaf image to get started.")
