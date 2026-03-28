# KrishiRakshak — AI-Powered Crop Disease Diagnosis System

## Project Identity
**KrishiRakshak** is a production-grade, end-to-end AI system for crop disease diagnosis with regional language support. It uses a fine-tuned Vision-Language Model (Florence-2) to diagnose crop diseases from leaf images and generate treatment advice, with Sarvam AI for Indian language translation and text-to-speech.

**This is NOT a tutorial project.** It is architected as a production system with CI/CD, monitoring, IaC, containerization, model versioning, and feedback loops.

---

## Architecture Overview

```
User (Streamlit/Mobile)
    │
    ▼
API Gateway (REST)
    │
    ▼
Lambda / FastAPI on ECS
    │
    ├──► S3 (store uploaded image)
    │
    ├──► SageMaker Endpoint (Florence-2 fine-tuned)
    │       └── Input: leaf image + prompt "Identify crop disease and suggest treatment"
    │       └── Output: JSON {disease, confidence, treatment_en}
    │
    ├──► Sarvam Mayura API (translate treatment to regional language)
    │
    ├──► Sarvam Bulbul API (text-to-speech in regional language)
    │
    └──► DynamoDB (log prediction + feedback for retraining)
```

---

## Model Strategy

### Primary: Microsoft Florence-2-base-ft (232M params)
- **Why:** Vision-Language Model — sees leaf image AND generates diagnosis text in one forward pass
- **Fine-tuning:** LoRA on PlantVillage dataset (38 disease classes, ~54K images)
- **Format:** Image → text prompt → text output (seq2seq)
- **Hosting:** SageMaker Real-time Endpoint (ml.g4dn.xlarge) or Serverless
- **HuggingFace ID:** `microsoft/Florence-2-base-ft`

### Fallback: Amazon Bedrock Claude Sonnet (for complex treatment advice)
- When Florence-2's generated treatment text is too generic
- Bedrock augments with India-specific pesticide names, dosages, and seasonal advice
- This is optional — Florence-2 handles most cases after fine-tuning

### Translation & Voice: Sarvam AI
- **Mayura v1:** Translation (en → hi/mr/ta/te/kn/bn)
- **Bulbul v1:** Text-to-Speech with Indian voices
- **Free tier:** ₹1,000 credits on signup, never expire

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Vision Model | Florence-2-base-ft (0.23B) | Small VLM, fine-tunable on T4, MIT license |
| Fine-tuning | LoRA via PEFT + HuggingFace Transformers | Memory efficient, <1hr on g4dn.xlarge |
| Model Hosting | SageMaker Real-time/Serverless Endpoint | Managed, auto-scaling, pay-per-use |
| Fallback LLM | Amazon Bedrock (Claude Sonnet) | Complex treatment advice augmentation |
| Translation | Sarvam Mayura API | Indic-native, colloquial-aware |
| TTS | Sarvam Bulbul API | Indian voices, regional accents |
| Backend API | FastAPI on ECS Fargate OR Lambda | Container-based for production |
| API Gateway | AWS API Gateway | Throttling, auth, CORS |
| Frontend | Streamlit (demo) / React (production) | Streamlit for demo, React for prod |
| Storage | S3 (images) + DynamoDB (logs/feedback) | Serverless, cheap |
| IaC | Terraform | Reproducible infrastructure |
| CI/CD | GitHub Actions → ECR → ECS | Automated deployment |
| Monitoring | CloudWatch + custom metrics dashboard | Latency, error rate, model confidence |
| Containerization | Docker | Consistent environments |
| Model Registry | SageMaker Model Registry | Version models, A/B testing |
| Secrets | AWS Secrets Manager | Sarvam API key, model configs |

---

## Dataset

### PlantVillage (Primary)
- **Source:** Kaggle — `emmarex/plantdisease` or `abdallahalidev/plantvillage-dataset`
- **Size:** ~54,000 images, 38 classes (14 crop species)
- **Classes include:** Tomato Early Blight, Tomato Late Blight, Potato Late Blight, Apple Scab, Grape Black Rot, Healthy variants, etc.
- **Format for Florence-2:** Each image paired with text: `<DISEASE_DIAGNOSIS> {disease_name}. Treatment: {treatment_text}`

### Treatment Knowledge Base
- Curated JSON/CSV with 38 entries mapping disease → treatment
- India-specific: local pesticide brand names, dosage in metric, availability info
- Source: ICAR (Indian Council of Agricultural Research) guidelines + Krishi Vigyan Kendra advisories

---

## Project Structure

```
krishirakshak/
├── CLAUDE.md                          # THIS FILE — master instructions
├── README.md                          # Project readme for GitHub
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project metadata
│
├── docs/
│   ├── ARCHITECTURE.md                # Detailed architecture decisions
│   ├── MODELS.md                      # Model selection rationale & benchmarks
│   ├── PRODUCTION.md                  # Production readiness checklist
│   ├── API_SPEC.md                    # OpenAPI specification
│   └── DEMO_SCRIPT.md                # 1-hour demo script for students
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app entry point
│   │   ├── routes.py                  # /diagnose, /health, /feedback endpoints
│   │   ├── schemas.py                 # Pydantic models for request/response
│   │   └── middleware.py              # CORS, logging, rate limiting
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── florence_inference.py      # Florence-2 inference wrapper
│   │   ├── bedrock_fallback.py        # Bedrock augmentation (optional)
│   │   └── model_config.py           # Model paths, thresholds, class mappings
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── sarvam_translate.py        # Sarvam Mayura translation service
│   │   ├── sarvam_tts.py             # Sarvam Bulbul TTS service
│   │   ├── s3_service.py             # Image upload/retrieval
│   │   └── feedback_service.py       # DynamoDB feedback logging
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── diagnosis_pipeline.py      # Orchestrator: image → diagnosis → translate → TTS
│   │   ├── image_preprocessor.py     # Resize, normalize, validate input images
│   │   └── response_builder.py       # Format final response with citations
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py                # Custom CloudWatch metrics (latency, confidence, errors)
│   │   ├── drift_detector.py         # Basic data drift detection on incoming images
│   │   └── alerts.py                 # SNS alerting for anomalies
│   │
│   └── tests/
│       ├── __init__.py
│       ├── test_api.py               # API endpoint tests
│       ├── test_pipeline.py          # Pipeline integration tests
│       ├── test_florence.py          # Model inference tests
│       ├── test_sarvam.py            # Sarvam API integration tests
│       └── conftest.py               # Shared fixtures
│
├── training/
│   ├── fine_tune_florence.py          # LoRA fine-tuning script for Florence-2
│   ├── prepare_dataset.py            # PlantVillage → Florence-2 format converter
│   ├── treatment_kb.json             # Disease → treatment knowledge base
│   ├── evaluate_model.py             # Evaluate fine-tuned vs base model
│   ├── push_to_sagemaker.py          # Package & upload model to SageMaker
│   └── training_config.yaml          # Hyperparameters, paths, LoRA config
│
├── infrastructure/
│   ├── terraform/
│   │   ├── main.tf                   # Root terraform config
│   │   ├── variables.tf              # Input variables
│   │   ├── outputs.tf                # Output values (endpoints, ARNs)
│   │   ├── s3.tf                     # S3 buckets
│   │   ├── dynamodb.tf               # DynamoDB tables
│   │   ├── sagemaker.tf              # SageMaker endpoint config
│   │   ├── ecs.tf                    # ECS Fargate service
│   │   ├── api_gateway.tf            # API Gateway
│   │   ├── iam.tf                    # IAM roles and policies
│   │   ├── secrets.tf                # Secrets Manager
│   │   ├── monitoring.tf             # CloudWatch dashboards & alarms
│   │   └── ecr.tf                    # ECR repository
│   │
│   └── docker/
│       ├── Dockerfile.api            # FastAPI container
│       ├── Dockerfile.training       # Training container for SageMaker
│       └── docker-compose.yml        # Local development setup
│
├── scripts/
│   ├── setup_aws.sh                  # One-time AWS setup (create roles, enable Bedrock)
│   ├── deploy.sh                     # Full deployment script
│   ├── warmup_endpoint.sh            # Warm up SageMaker endpoint before demo
│   ├── cleanup.sh                    # Tear down all resources (save credits)
│   ├── download_dataset.sh           # Download PlantVillage from Kaggle
│   └── run_local.sh                  # Run locally for development
│
├── configs/
│   ├── app_config.yaml               # Application configuration
│   ├── model_config.yaml             # Model thresholds, class names
│   ├── sarvam_config.yaml            # Sarvam API config, language codes
│   └── monitoring_config.yaml        # Alert thresholds, metric namespaces
│
├── frontend/
│   ├── streamlit_app.py              # Streamlit demo UI
│   ├── components/
│   │   ├── image_uploader.py         # Image upload widget with preview
│   │   ├── results_display.py        # Diagnosis results panel
│   │   ├── audio_player.py           # Audio playback for TTS output
│   │   ├── feedback_widget.py        # Thumbs up/down feedback
│   │   └── language_selector.py      # Language dropdown
│   └── assets/
│       └── logo.png                  # KrishiRakshak logo
│
└── .github/
    └── workflows/
        ├── ci.yml                    # Lint, test, build on PR
        ├── cd.yml                    # Deploy to ECS on merge to main
        └── model_retrain.yml         # Scheduled model retraining pipeline
```

---

## Key Implementation Rules

### 1. Florence-2 Fine-tuning
- Use `microsoft/Florence-2-base-ft` (NOT large — budget constraint)
- Fine-tune with LoRA (rank=8, alpha=16) using PEFT library
- Custom task prompt: `<CROP_DISEASE>` → model generates `{disease_name}. Treatment: {advice}`
- Training: 10 epochs, batch_size=8, lr=1e-4, on ml.g4dn.xlarge (~$5-8)
- Freeze image encoder, train only decoder + LoRA adapters
- Save adapter weights separately (small, fast to deploy)

### 2. Production Patterns
- **Health check endpoint:** `/health` returns model status, version, uptime
- **Request validation:** Reject non-image files, oversized uploads (>5MB), malformed requests
- **Graceful degradation:** If Sarvam API fails → return English only. If SageMaker cold → return "processing" with retry. If confidence <40% → return "uncertain, consult expert"
- **Idempotency:** Each request gets a unique ID, stored in DynamoDB
- **Rate limiting:** 100 requests/minute per API key
- **Versioned API:** `/v1/diagnose`

### 3. Monitoring
- **Model metrics:** Inference latency (p50, p95, p99), confidence distribution, class distribution
- **Business metrics:** Requests per day, language breakdown, feedback ratio (positive/negative)
- **Alerts:** Confidence <40% for >10% of requests → alert. Latency p99 >10s → alert. Error rate >5% → alert.
- **Drift detection:** Compare incoming image brightness/size distribution against training data baseline

### 4. Security
- Sarvam API key in Secrets Manager, NEVER in code
- API Gateway with API key authentication
- S3 bucket policy: no public access
- IAM least-privilege roles per service
- Input sanitization: validate image format, strip EXIF data

### 5. Testing
- Unit tests for each service (mock external APIs)
- Integration tests for full pipeline
- Model tests: assert accuracy >90% on test set, latency <3s
- Load test: locust script for 50 concurrent users

---

## Build Order (for Claude Code)

Execute in this exact sequence:

### Phase 1: Foundation
1. `requirements.txt` and `pyproject.toml`
2. `configs/` — all YAML configs
3. `training/treatment_kb.json` — disease-treatment knowledge base
4. `scripts/download_dataset.sh`

### Phase 2: Model Training
5. `training/prepare_dataset.py` — convert PlantVillage to Florence-2 format
6. `training/fine_tune_florence.py` — LoRA fine-tuning script
7. `training/evaluate_model.py` — compare base vs fine-tuned
8. `training/push_to_sagemaker.py` — package for deployment

### Phase 3: Backend Services
9. `src/models/florence_inference.py`
10. `src/models/bedrock_fallback.py`
11. `src/services/sarvam_translate.py`
12. `src/services/sarvam_tts.py`
13. `src/services/s3_service.py`
14. `src/services/feedback_service.py`
15. `src/pipeline/diagnosis_pipeline.py`
16. `src/pipeline/image_preprocessor.py`

### Phase 4: API Layer
17. `src/api/schemas.py`
18. `src/api/routes.py`
19. `src/api/middleware.py`
20. `src/api/main.py`

### Phase 5: Frontend
21. `frontend/streamlit_app.py` and components

### Phase 6: Infrastructure
22. `infrastructure/docker/Dockerfile.api`
23. `infrastructure/docker/Dockerfile.training`
24. `infrastructure/terraform/*.tf`

### Phase 7: CI/CD & Monitoring
25. `src/monitoring/metrics.py`
26. `src/monitoring/drift_detector.py`
27. `.github/workflows/*.yml`
28. `scripts/*.sh`

### Phase 8: Tests
29. All test files in `src/tests/`

### Phase 9: Documentation
30. All files in `docs/`
31. `README.md`

---

## Cost Budget: $250

| Item | Estimated Cost |
|------|---------------|
| SageMaker Training (g4dn.xlarge, 2-3 hrs) | $5-8 |
| SageMaker Endpoint (demo hours) | $5-15 |
| Bedrock calls (optional, ~100 calls) | $3-5 |
| ECR + ECS Fargate (demo hours) | $5-10 |
| S3 + DynamoDB | $1 |
| Secrets Manager | $0.40 |
| Sarvam AI | ₹0 (free tier) |
| **Total** | **$20-40** |
| **Remaining for student experiments** | **~$210** |

---

## Critical Reminders
- Florence-2 requires `trust_remote_code=True` when loading
- PlantVillage dataset: some classes are highly imbalanced — apply class weights or oversample
- Sarvam API rate limits on free tier — add 200ms delay between calls
- SageMaker cold start: warm endpoint 5 min before demo
- All configs externalized in YAML — no hardcoded values in source code
- Every function must have error handling — this is production code
- Use structured logging (JSON format) throughout — not print statements
