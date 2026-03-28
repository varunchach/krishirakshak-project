# Demo Script — KrishiRakshak (1 Hour)

## Pre-Demo Setup (30 min before)
1. Warm up SageMaker endpoint: `bash scripts/warmup_endpoint.sh`
2. Open Streamlit app: `streamlit run frontend/streamlit_app.py`
3. Keep 5 test images ready (3 diseased, 1 healthy, 1 non-leaf)
4. Open Neo4j/SageMaker/CloudWatch consoles in browser tabs
5. Test Sarvam API with one call to confirm credits are active

---

## Minute 0-8: The Problem (Hook the audience)
**Key message:** "Farmers lose ₹90,000 crore annually to crop diseases. Manual inspection doesn't scale."

- Show real crop loss statistics for India (ICAR data)
- Show a diseased tomato leaf photo — ask students "What disease is this?"
- Nobody will know. That's the point.
- "Now imagine a farmer in Nashik with no access to an extension officer. What does he do?"

## Minute 8-18: Architecture (Teach the thinking)
**Key message:** "We chose best-in-class for each component, not forced everything into one cloud."

- Draw the architecture on whiteboard/slide:
  - Florence-2 (why VLM, not classifier + LLM)
  - Sarvam AI (why not Amazon Translate)
  - SageMaker (why managed endpoint, not raw EC2)
- **Critical slide:** Show the "Demo vs Production" comparison table from docs/PRODUCTION.md
- Ask students: "What's missing if I just run `streamlit run app.py`?"
  - Monitoring, CI/CD, error handling, auth, feedback loop, IaC

## Minute 18-22: Florence-2 Deep Dive (The model choice)
**Key message:** "Your students used EfficientNet for classification. That's 2020. This is 2026."

- Explain: Florence-2 sees image AND generates text in ONE forward pass
- Show the architecture diagram: DaViT encoder → Transformer decoder
- Show model size comparison: Florence-2 (232M) vs LLaVA (7B) vs GPT-4V (??B)
- "We fine-tuned it with LoRA — only 10MB of adapter weights on top of 232M base"
- Show the training config YAML — students see actual hyperparameters

## Minute 22-38: Live Demo (The wow moments)
**Moment 1:** Upload diseased tomato leaf → See diagnosis + treatment in English
- Point out: "One API call. One model. Image in, structured text out."

**Moment 2:** Switch language to Hindi → Treatment appears in Hindi
- "Sarvam Mayura — trained specifically for Indian languages"

**Moment 3:** Hit PLAY on audio → Treatment spoken aloud in Hindi
- "Sarvam Bulbul — Indian voice. This is what a farmer hears."
- THIS IS YOUR PEAK MOMENT. Let it land. Pause.

**Moment 4:** Upload a healthy leaf → "Your crop is healthy!"
- Show the confidence threshold logic

**Moment 5:** Upload a random non-leaf image (e.g., your hand) → Graceful error
- "Production systems handle bad input. They don't crash."

## Minute 38-48: Code Walkthrough (Show the craft)
- Walk through `src/pipeline/diagnosis_pipeline.py` — the orchestrator
- Show error handling chain (SageMaker fail → Sarvam fail → graceful degrade)
- Show `configs/app_config.yaml` — "No hardcoded values. Everything is config."
- Show `infrastructure/terraform/main.tf` — "Infrastructure as code. One command to deploy."
- Show `.github/workflows/ci.yml` — "Every PR is tested automatically."

## Minute 48-55: Production Narrative (The real lesson)
**Key message:** "The model is 10% of the work. Production engineering is the other 90%."

- Show CloudWatch dashboard (even if demo-scale data)
- Explain the feedback loop: farmer says "wrong" → logged → triggers retrain evaluation
- Explain model versioning: v1.0 → v1.1 → A/B test → promote winner
- Explain drift detection: "What if camera phones change and images look different?"
- **Be honest:** "Marathi agricultural terms aren't perfect in translation. This is a known limitation. In production, you'd build a glossary override."

## Minute 55-60: Q&A
**Expected questions and honest answers:**
- "Why not just use GPT-4V?" → "Cost. GPT-4V = $0.01/image. Florence-2 on SageMaker = $0.001/image at scale."
- "Does this work for Indian crops not in PlantVillage?" → "No. You'd need to collect Indian crop data and retrain. That's the next version."
- "Can this run on a phone?" → "Not yet. But Florence-2 at 232M can be quantized to INT4 and run on-device. That's the roadmap."
- "How accurate is it really?" → Show test set metrics honestly. Don't inflate.

---

## Emergency Fallbacks
- SageMaker endpoint down → Have pre-recorded demo video as backup
- Sarvam API down → Show translation/TTS from a cached response
- Internet down → Have the Streamlit app running against a local model on laptop
- Student asks a question you don't know → "Great question. I'll look into it and share on the group." NEVER bluff.
