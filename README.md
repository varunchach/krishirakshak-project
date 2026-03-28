# 🌾 KrishiRakshak — AI-Powered Crop Disease Diagnosis

> An end-to-end, production-grade AI system for diagnosing crop diseases from leaf images with regional Indian language support.

## What This Is
KrishiRakshak uses a fine-tuned **Microsoft Florence-2** Vision-Language Model to diagnose 38 crop diseases from a single leaf photo and generate treatment recommendations in Hindi, Marathi, Tamil, Telugu, and other Indian languages — including voice output.

## What Makes This Different
- **Not a classifier.** Florence-2 is a VLM — it sees the image AND generates text in one pass.
- **Not a tutorial.** Production architecture: Docker, Terraform, CI/CD, monitoring, feedback loops.
- **Not English-only.** Sarvam AI for Indian language translation + text-to-speech.
- **Not a notebook.** Deployed as a containerized FastAPI service on AWS.

## Architecture
```
Leaf Image → SageMaker (Florence-2) → Disease + Treatment (EN)
                                        ↓
                                   Sarvam Mayura → Translation (HI/MR/TA/TE)
                                        ↓
                                   Sarvam Bulbul → Voice Output (WAV)
```

## Tech Stack
| Component | Technology |
|-----------|-----------|
| Vision Model | Florence-2-base-ft (232M params, LoRA fine-tuned) |
| Model Hosting | AWS SageMaker |
| Translation | Sarvam Mayura API |
| Text-to-Speech | Sarvam Bulbul API |
| Backend | FastAPI on ECS Fargate |
| Frontend | Streamlit (demo) |
| IaC | Terraform |
| CI/CD | GitHub Actions |
| Monitoring | CloudWatch + custom metrics |

## Quick Start

### Prerequisites
- AWS account with $250 credits
- Sarvam AI account (free ₹1,000 credits on signup)
- Python 3.9+
- Docker
- Terraform

### Setup
```bash
# Clone and install
git clone https://github.com/your-username/krishirakshak.git
cd krishirakshak
pip install -r requirements.txt

# Download dataset
bash scripts/download_dataset.sh

# Configure AWS
aws configure
bash scripts/setup_aws.sh

# Fine-tune model
python training/prepare_dataset.py
python training/fine_tune_florence.py

# Deploy
bash scripts/deploy.sh

# Run demo
streamlit run frontend/streamlit_app.py
```

## Cost
Total demo cost: ~$20-40 out of $250 budget.

## Documentation
- [Architecture Decisions](docs/ARCHITECTURE.md)
- [Model Selection](docs/MODELS.md)
- [Production Readiness](docs/PRODUCTION.md)
- [API Specification](docs/API_SPEC.md)
- [Demo Script](docs/DEMO_SCRIPT.md)

## License
MIT
