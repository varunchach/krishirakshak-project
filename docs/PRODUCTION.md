# Production Readiness — KrishiRakshak

## Checklist

### Infrastructure
- [ ] All resources provisioned via Terraform (no console clicks)
- [ ] Separate dev/staging/prod environments (even if staging is minimal)
- [ ] Secrets in AWS Secrets Manager (Sarvam API key, model config)
- [ ] S3 buckets: no public access, versioning enabled, lifecycle policy
- [ ] DynamoDB: on-demand capacity, point-in-time recovery enabled
- [ ] ECR: image scanning enabled, lifecycle policy for old images

### API
- [ ] Versioned endpoints (`/v1/diagnose`)
- [ ] OpenAPI spec documented (auto-generated from FastAPI)
- [ ] Health check endpoint (`/health`) returns model version, uptime, dependencies status
- [ ] Request validation: image format, size limits (max 5MB), content-type checks
- [ ] Rate limiting: 100 req/min per API key via API Gateway
- [ ] CORS configured for frontend domain only
- [ ] Request ID on every request (UUID), returned in response headers
- [ ] Structured JSON logging (timestamp, request_id, level, message, latency_ms)

### Model Serving
- [ ] SageMaker endpoint with auto-scaling policy (min=1, max=3)
- [ ] Model registered in SageMaker Model Registry with version tag
- [ ] A/B testing capability via production variants (weight split)
- [ ] Cold start mitigation: periodic health pings OR minimum instance count
- [ ] Model artifact stored in S3 with versioned path: `s3://models/florence2/v{version}/model.tar.gz`
- [ ] Inference container tested locally before SageMaker deployment

### Resilience
- [ ] Graceful degradation chain:
      - SageMaker down → return "service temporarily unavailable, try again"
      - Sarvam down → return English-only result (skip translation/TTS)
      - Bedrock down → skip augmentation, use Florence-2 output only
      - DynamoDB down → skip feedback logging, don't block response
- [ ] Retry logic: 3 retries with exponential backoff for Sarvam API calls
- [ ] Circuit breaker pattern for external API dependencies
- [ ] Dead letter queue (SQS) for failed feedback writes
- [ ] Timeout configuration: SageMaker=30s, Sarvam=10s, total request=45s

### Monitoring & Alerting
- [ ] CloudWatch dashboard with:
      - Request count / error rate / latency (p50, p95, p99)
      - Model confidence distribution (histogram)
      - Language breakdown (pie chart)
      - Feedback ratio (positive vs negative)
- [ ] Alarms configured:
      - Error rate >5% for 5 minutes → SNS email
      - Latency p99 >10s → SNS email
      - SageMaker endpoint unhealthy → SNS email
      - Billing alarm at $50 threshold
- [ ] X-Ray tracing enabled for request flow visualization
- [ ] Log retention: 30 days in CloudWatch, archive to S3 after

### CI/CD
- [ ] GitHub Actions pipeline:
      - On PR: lint (ruff), type check (mypy), unit tests (pytest)
      - On merge to main: build Docker image → push to ECR → deploy to ECS
- [ ] Docker image tagged with git commit SHA (not just `latest`)
- [ ] Rollback capability: previous task definition always available
- [ ] Model retraining pipeline: scheduled or triggered by feedback threshold

### Security
- [ ] IAM roles: least privilege per service
- [ ] API Gateway: API key required for all endpoints
- [ ] Input sanitization: strip EXIF metadata from images (privacy)
- [ ] No PII logged (farmer location, device info anonymized)
- [ ] Dependency scanning: `pip audit` in CI pipeline
- [ ] Container: non-root user in Dockerfile

### Data Pipeline (Feedback Loop)
- [ ] Every prediction logged to DynamoDB: request_id, image_key, prediction, confidence, timestamp
- [ ] Feedback endpoint: farmer/user submits correct/incorrect + optional comment
- [ ] Weekly export: DynamoDB → S3 (Parquet format)
- [ ] Retraining trigger: when negative feedback >30% over 7 days
- [ ] Retrained model goes through evaluation → approval → deployment cycle
- [ ] Old model kept as fallback in Model Registry

### Documentation
- [ ] README with setup instructions
- [ ] API documentation (auto from FastAPI /docs)
- [ ] Architecture decision records (docs/ARCHITECTURE.md)
- [ ] Runbook: common issues and how to fix them
- [ ] Demo script for 1-hour presentation

---

## What Makes This "Production" vs "Demo"

| Aspect | Demo | Production |
|--------|------|------------|
| Deployment | `streamlit run app.py` | Docker → ECR → ECS Fargate |
| Infrastructure | Console clicks | Terraform |
| Monitoring | Print statements | CloudWatch + X-Ray + custom metrics |
| Error handling | Try/except pass | Graceful degradation chain |
| Config | Hardcoded strings | YAML configs + Secrets Manager |
| Testing | Manual | Automated pytest + CI |
| Logging | print() | Structured JSON logging |
| Security | None | API keys + IAM + input validation |
| Model updates | Re-upload manually | Model Registry + A/B deploy |
| Feedback | None | DynamoDB + retraining pipeline |

This table is the key slide in your demo. Show it.
