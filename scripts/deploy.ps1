# KrishiRakshak — Build, push to ECR, redeploy ECS
# Run from project root in PowerShell: .\scripts\deploy.ps1

$ErrorActionPreference = "Stop"

$REGION   = "us-east-1"
$ACCOUNT  = "593755927741"
$REGISTRY = "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
$REPO     = "$REGISTRY/krishirakshak-api"
$TAG      = git rev-parse --short HEAD

Write-Host "==> [1/4] Writing ECR credentials..." -ForegroundColor Cyan
python scripts\ecr_auth.py
Write-Host "    Done" -ForegroundColor Green

Write-Host "==> [2/4] Building Docker image (tag=$TAG)..." -ForegroundColor Cyan
Write-Host "    First build ~5-10 min..." -ForegroundColor Yellow
docker build -t "${REPO}:${TAG}" -t "${REPO}:latest" .
Write-Host "    Build OK" -ForegroundColor Green

Write-Host "==> [3/4] Pushing to ECR..." -ForegroundColor Cyan
docker push "${REPO}:${TAG}"
docker push "${REPO}:latest"
Write-Host "    Push OK" -ForegroundColor Green

Write-Host "==> [4/4] ECS redeployment..." -ForegroundColor Cyan
python -c "import boto3; boto3.client('ecs',region_name='us-east-1').update_service(cluster='krishirakshak-cluster',service='krishirakshak-api',forceNewDeployment=True); print('Triggered')"
Write-Host "    Done" -ForegroundColor Green

Write-Host ""
Write-Host "=== Deploy complete ===" -ForegroundColor Green
Write-Host "API URL : https://sgy86f7n6l.execute-api.us-east-1.amazonaws.com"
Write-Host "Health  : https://sgy86f7n6l.execute-api.us-east-1.amazonaws.com/v1/health"
