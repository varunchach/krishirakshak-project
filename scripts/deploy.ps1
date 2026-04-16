# KrishiRakshak — Build, push to ECR, redeploy ECS (API + UI)
# Run from project root in PowerShell: .\scripts\deploy.ps1

$ErrorActionPreference = "Stop"

$REGION    = "us-east-1"
$ACCOUNT   = "593755927741"
$REGISTRY  = "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
$REPO_API  = "$REGISTRY/krishirakshak-api"
$REPO_UI   = "$REGISTRY/krishirakshak-ui"
$TAG       = git rev-parse --short HEAD

Write-Host "==> [1/6] Writing ECR credentials..." -ForegroundColor Cyan
python scripts\ecr_auth.py
Write-Host "    Done" -ForegroundColor Green

# ── API (FastAPI) ──────────────────────────────────────────────────────────────
Write-Host "==> [2/6] Building API image (tag=$TAG)..." -ForegroundColor Cyan
Write-Host "    First build ~5-10 min..." -ForegroundColor Yellow
docker build -f Dockerfile -t "${REPO_API}:${TAG}" -t "${REPO_API}:latest" .
Write-Host "    Build OK" -ForegroundColor Green

Write-Host "==> [3/6] Pushing API image to ECR..." -ForegroundColor Cyan
docker push "${REPO_API}:${TAG}"
docker push "${REPO_API}:latest"
Write-Host "    Push OK" -ForegroundColor Green

# ── UI (Streamlit) ─────────────────────────────────────────────────────────────
Write-Host "==> [4/6] Building UI image (tag=$TAG)..." -ForegroundColor Cyan
docker build -f Dockerfile.streamlit -t "${REPO_UI}:${TAG}" -t "${REPO_UI}:latest" .
Write-Host "    Build OK" -ForegroundColor Green

Write-Host "==> [5/6] Pushing UI image to ECR..." -ForegroundColor Cyan
docker push "${REPO_UI}:${TAG}"
docker push "${REPO_UI}:latest"
Write-Host "    Push OK" -ForegroundColor Green

# ── ECS redeployment ───────────────────────────────────────────────────────────
Write-Host "==> [6/6] Triggering ECS redeployment (API + UI)..." -ForegroundColor Cyan
python -c "
import boto3
ecs = boto3.client('ecs', region_name='us-east-1')
ecs.update_service(cluster='krishirakshak-cluster', service='krishirakshak-api', forceNewDeployment=True)
ecs.update_service(cluster='krishirakshak-cluster', service='krishirakshak-ui',  forceNewDeployment=True)
print('Both services redeployment triggered')
"
Write-Host "    Done" -ForegroundColor Green

Write-Host ""
Write-Host "=== Deploy complete ===" -ForegroundColor Green
Write-Host "API URL : https://sgy86f7n6l.execute-api.us-east-1.amazonaws.com"
Write-Host "Health  : https://sgy86f7n6l.execute-api.us-east-1.amazonaws.com/v1/health"
Write-Host "UI URL  : Check provisioning output for UI ALB DNS"
