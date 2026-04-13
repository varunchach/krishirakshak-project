# KrishiRakshak — Agentic AI for Indian Farmers

Crop disease detection + Hindi/English document Q&A + audio responses.

---

## Architecture

```
User (image / text / PDF)
        ↓
  Streamlit UI  (localhost:8501)
        ↓
  FastAPI (localhost:8000)
        ↓
  LangGraph ReAct Agent
    ├── EfficientNet-B3   → SageMaker (ap-south-1)  krishirakshak-efficientnet-b3
    ├── BGE-M3 Embeddings → SageMaker (ap-south-1)  bge-m3-krishirakshak
    ├── Claude Sonnet 4.6 → Bedrock (us-east-1)
    ├── FAISS + BM25 hybrid retrieval
    └── Amazon Polly TTS
        ↓
  DynamoDB  → krishirakshak-predictions-dev
  S3        → krishirakshak-assets-dev
```

---

## What Is Done

### Models and AWS Resources
- [x] EfficientNet-B3 trained on 5 diseases, deployed to SageMaker (`krishirakshak-efficientnet-b3`, ap-south-1, InService)
- [x] BGE-M3 multilingual embeddings deployed to SageMaker (`bge-m3-krishirakshak`, ap-south-1, InService)
- [x] S3 bucket created (`krishirakshak-assets-dev`) with 5 disease test images uploaded
- [x] DynamoDB table created (`krishirakshak-predictions-dev`)
- [x] ECR repository created (`593755927741.dkr.ecr.us-east-1.amazonaws.com/krishirakshak-api`)
- [x] ECS cluster + service provisioned (`krishirakshak-cluster`)
- [x] Internal ALB + API Gateway + VPC link provisioned
- [x] API URL: `https://sgy86f7n6l.execute-api.us-east-1.amazonaws.com`
- [x] IAM roles for ECS task + execution

### Application Code
- [x] FastAPI with `/v1/diagnose`, `/v1/query`, `/v1/ingest`, `/v1/feedback`, `/v1/health`
- [x] EfficientNet-B3 dual backend (sagemaker / local)
- [x] BGE-M3 embeddings via SageMaker
- [x] FAISS + BM25 hybrid search with RRF
- [x] LangGraph ReAct agent with MemorySaver (session-aware)
- [x] Amazon Polly TTS (English: Joanna, Hindi: Aditi)
- [x] Hindi/English language detection (Lingua)
- [x] PDF ingestion + chunking
- [x] DynamoDB prediction logging + feedback service
- [x] S3 service for image + audio upload
- [x] CloudWatch metrics + SNS alerts

### Infrastructure Scripts
- [x] `scripts/provision.py` — provisions all AWS infrastructure via boto3 (no Terraform)
- [x] `scripts/deploy.ps1` — builds Docker image, pushes to ECR, redeploys ECS
- [x] `scripts/ecr_auth.py` — writes ECR credentials to ~/.docker/config.json
- [x] `scripts/cleanup.sh` — tears down all AWS resources
- [x] `scripts/ingest_docs.sh` — ingests PDFs into FAISS via API
- [x] `efficientnet-deploy/` — SageMaker deployment for EfficientNet-B3
- [x] `Dockerfile` + `requirements.prod.txt` — production Docker image (~500MB, no torch)

---

## What Is Pending

### Step 4 — Docker Build and ECS Deploy (BLOCKED — see issue below)
- [ ] Build Docker image and push to ECR
- [ ] ECS service pulls image and starts

### Step 5 — Chat UI
- [ ] Rewrite `frontend/streamlit_app.py` with `st.chat_message` session-aware interface
- [ ] Image upload + text query + audio playback in one UI

### Step 6 — PDF Ingestion
- [ ] Run `bash scripts/ingest_docs.sh` against live API to populate FAISS index

---

## Known Issue — ECR Login Hangs in PowerShell

**Symptom:** `python scripts\ecr_auth.py` hangs at `step2: calling ECR...` in PowerShell.

**Not affected:** Git Bash — ECR works fine there (all services confirmed OK).

**Root cause:** Unknown. STS, S3, ECS all respond in ~5s from PowerShell but ECR hangs indefinitely. Happens with both root credentials and IAM user credentials.

**Things tried:**
- `docker login --password-stdin` (hangs)
- `docker login --password` directly (hangs)
- Python Docker SDK `client.login()` (hangs)
- Writing auth to `~/.docker/config.json` directly via Python (hangs at boto3 ECR call)
- New IAM user with AdministratorAccess (still hangs)
- Fresh PowerShell window (still hangs)

**To try next session:**
1. Try from WSL if installed
2. Check if Windows Defender or antivirus is intercepting ECR HTTPS traffic
3. Try disabling VPN if active
4. Try from a different machine or terminal (VS Code terminal, cmd.exe)
5. Test: `python -c "import boto3; boto3.client('ecr',region_name='us-east-1').get_authorization_token(); print('OK')"` from cmd.exe (not PowerShell)

---

## Running Locally (Without Docker)

```powershell
pip install -r requirements.txt

$env:CLASSIFIER_BACKEND            = "sagemaker"
$env:CLASSIFIER_SAGEMAKER_REGION   = "ap-south-1"
$env:CLASSIFIER_SAGEMAKER_ENDPOINT = "krishirakshak-efficientnet-b3"
$env:SAGEMAKER_REGION              = "ap-south-1"
$env:SAGEMAKER_ENDPOINT            = "bge-m3-krishirakshak"
$env:S3_BUCKET                     = "krishirakshak-assets-dev"
$env:DYNAMODB_TABLE                = "krishirakshak-predictions-dev"
$env:AWS_DEFAULT_REGION            = "us-east-1"

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs
Streamlit UI: http://localhost:8501

---

## Supported Diseases

| Disease | Crop |
|---|---|
| Tomato Early Blight | Tomato |
| Tomato Late Blight | Tomato |
| Potato Late Blight | Potato |
| Tomato Leaf Mold | Tomato |
| Corn Common Rust | Corn |

---

## Key Config

| Variable | Value |
|---|---|
| `CLASSIFIER_BACKEND` | `sagemaker` (prod) / `local` (dev) |
| `CLASSIFIER_SAGEMAKER_ENDPOINT` | `krishirakshak-efficientnet-b3` |
| `SAGEMAKER_ENDPOINT` | `bge-m3-krishirakshak` |
| `S3_BUCKET` | `krishirakshak-assets-dev` |
| `DYNAMODB_TABLE` | `krishirakshak-predictions-dev` |
| `AWS_DEFAULT_REGION` | `us-east-1` |
| `SAGEMAKER_REGION` | `ap-south-1` |
