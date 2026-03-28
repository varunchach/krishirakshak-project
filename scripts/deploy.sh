#!/bin/bash
# Deploy KrishiRakshak to AWS
set -euo pipefail

PROJECT="krishirakshak"
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$PROJECT-api"

echo "=== Deploying KrishiRakshak ==="

# 1. Terraform
echo "Provisioning infrastructure..."
cd infrastructure/terraform
terraform init
terraform apply -auto-approve
cd ../..

# 2. Docker build & push
echo "Building Docker image..."
docker build -f infrastructure/docker/Dockerfile.api -t "$PROJECT-api:latest" .

echo "Pushing to ECR..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_REPO"
docker tag "$PROJECT-api:latest" "$ECR_REPO:latest"
docker push "$ECR_REPO:latest"

# 3. Deploy model (if not already deployed)
echo "Deploying model to SageMaker..."
python training/push_to_sagemaker.py

echo ""
echo "=== Deployment complete! ==="
echo "API: Check ECS service in console"
echo "Model: https://console.aws.amazon.com/sagemaker/home#/endpoints/$PROJECT-florence2"
