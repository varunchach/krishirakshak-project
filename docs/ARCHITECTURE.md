# Architecture Decisions — KrishiRakshak

## ADR-001: Florence-2 over EfficientNet + Separate LLM

### Context
Traditional approach: EfficientNet classifies disease → separate LLM generates treatment.
New approach: Florence-2 (VLM) does both in one forward pass.

### Decision
Use Florence-2-base-ft (232M params) as the primary model.

### Rationale
- **Single model inference** reduces latency by ~40% (one API call vs two)
- **End-to-end fine-tuning** means the model learns disease-specific treatment language
- **232M params** fits on a T4 GPU — 10x smaller than LLaVA (7B)
- **Production benefit:** One model to version, monitor, and retrain — not two
- **Student impact:** They've seen classification models. They've never seen a VLM fine-tuned for domain-specific image→text generation.

### Tradeoffs
- Florence-2's treatment text won't be as detailed as Claude Sonnet — mitigated by Bedrock fallback for complex cases
- Fine-tuning requires careful prompt engineering for the seq2seq format

---

## ADR-002: Sarvam AI over Amazon Translate

### Context
Need Hindi/Marathi/Telugu translation of treatment advice.

### Decision
Use Sarvam Mayura (translation) + Bulbul (TTS) APIs.

### Rationale
- Sarvam is trained specifically for Indian languages with colloquial awareness
- Amazon Translate is generic, weaker on agricultural Marathi/Telugu terms
- Sarvam Bulbul adds voice output — critical for farmer-facing apps (70%+ farmers prefer voice)
- Free tier (₹1,000) covers entire demo and experimentation
- Calling external API from Lambda/ECS is standard production pattern

### Tradeoffs
- Not 100% AWS-native — acceptable for best-in-class component selection
- Sarvam has rate limits on free tier — add request queuing for production

---

## ADR-003: ECS Fargate over Lambda for API

### Context
Lambda has 15-minute timeout and cold start. Florence-2 inference + Sarvam calls can take 10-15 seconds.

### Decision
Use ECS Fargate with FastAPI container for production. Lambda acceptable for demo-only.

### Rationale
- **Consistent latency:** No cold start with Fargate (minimum 1 task always running)
- **Model loading:** Florence-2 processor can be loaded once at container startup
- **Concurrent requests:** Fargate handles multiple requests natively
- **Docker-based:** Same container runs locally, in CI, and in production

### Tradeoffs
- Higher baseline cost than Lambda ($0.04/hr vs pay-per-invocation)
- For demo with low traffic, Lambda is cheaper — use Lambda for demo, Fargate for "production" narrative

---

## ADR-004: SageMaker Endpoint over Self-Hosted Inference

### Context
Florence-2 needs GPU for inference. Options: EC2 with GPU, SageMaker Endpoint, or embed in ECS.

### Decision
Deploy Florence-2 on SageMaker Real-time Endpoint.

### Rationale
- **Model versioning** via SageMaker Model Registry
- **Auto-scaling** built in
- **A/B testing** via production variants (e.g., base vs fine-tuned)
- **Monitoring** via SageMaker Model Monitor
- **Separation of concerns:** API layer (ECS) is decoupled from model serving (SageMaker)

### Tradeoffs
- SageMaker endpoints have minimum instance cost even when idle
- For demo: use on-demand, shut down immediately after

---

## ADR-005: DynamoDB for Feedback over RDS

### Context
Need to store farmer feedback (correct/incorrect diagnosis) for model retraining.

### Decision
DynamoDB with single-table design.

### Rationale
- **Free tier:** 25 GB storage, 25 WCU/RCU — more than enough
- **Serverless:** No connection pooling, no maintenance
- **Schema:** `PK: REQUEST#{request_id}`, `SK: METADATA | FEEDBACK | PREDICTION`
- **Export to S3:** DynamoDB export for batch retraining jobs

### Table Schema
```
PK                      SK              Attributes
REQUEST#uuid-123        METADATA        timestamp, image_s3_key, language, device
REQUEST#uuid-123        PREDICTION      disease, confidence, treatment_en, treatment_translated
REQUEST#uuid-123        FEEDBACK        is_correct (bool), farmer_comment, submitted_at
```

---

## ADR-006: Terraform over CloudFormation

### Context
Need Infrastructure as Code for reproducibility.

### Decision
Terraform with AWS provider.

### Rationale
- Industry standard (more hireable skill than CloudFormation)
- Multi-cloud portable if needed later
- Better state management and plan/apply workflow
- Richer module ecosystem
- Students learn a transferable skill

---

## ADR-007: Monitoring Strategy

### Layers
1. **Infrastructure:** CloudWatch for ECS, SageMaker, Lambda metrics
2. **Application:** Custom metrics pushed to CloudWatch (inference latency, confidence scores)
3. **Model:** Confidence distribution drift, class distribution shift
4. **Business:** Request volume, language breakdown, feedback ratio

### Alerting Rules
| Metric | Threshold | Action |
|--------|-----------|--------|
| Inference latency p99 | >10s | SNS alert to team |
| Average confidence | <40% for 10+ requests | Flag for model investigation |
| Error rate | >5% over 5 min | SNS alert + auto-scale |
| Feedback negative ratio | >30% over 24h | Trigger retraining review |
| Image size anomaly | >3 std from mean | Log for drift analysis |
