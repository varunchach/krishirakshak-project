#!/bin/bash
# Build Docker image, push to ECR, force ECS redeployment.
# Usage: bash scripts/deploy.sh [dev|prod]
#
# Prerequisites: Docker Desktop running

set -euo pipefail

ENV="${1:-dev}"
PROJECT="krishirakshak"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
ACCOUNT_ID=$(python -c "import boto3; print(boto3.client('sts').get_caller_identity()['Account'])")
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${PROJECT}-api"
IMAGE_TAG=$(git rev-parse --short HEAD)
API_URL="https://sgy86f7n6l.execute-api.us-east-1.amazonaws.com"

echo "==> Deploying KrishiRakshak API"
echo "    Environment : $ENV"
echo "    Image tag   : $IMAGE_TAG"
echo "    ECR repo    : $ECR_REPO"
echo ""

# 1. Login to ECR via boto3
echo "==> Logging in to ECR..."
ECR_PASSWORD=$(python -c "
import boto3, base64
token = boto3.client('ecr', region_name='${REGION}').get_authorization_token()
auth = token['authorizationData'][0]['authorizationToken']
print(base64.b64decode(auth).decode().split(':')[1])
")
echo "$ECR_PASSWORD" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# 2. Build image
echo "==> Building Docker image..."
docker build -t "${ECR_REPO}:${IMAGE_TAG}" -t "${ECR_REPO}:latest" .

# 3. Push to ECR
echo "==> Pushing to ECR..."
docker push "${ECR_REPO}:${IMAGE_TAG}"
docker push "${ECR_REPO}:latest"

# 4. Force ECS redeployment
echo "==> Triggering ECS redeployment..."
python -c "
import boto3
boto3.client('ecs', region_name='${REGION}').update_service(
    cluster='${PROJECT}-cluster',
    service='${PROJECT}-api',
    forceNewDeployment=True,
)
print('  ECS redeployment triggered')
"

echo ""
echo "==> Done. API URL: ${API_URL}"
echo "    Health check: curl ${API_URL}/v1/health"
echo ""
echo "==> Monitor ECS tasks (takes ~2 min to stabilize):"
echo "    python -c \"import boto3,json; r=boto3.client('ecs',region_name='${REGION}').describe_services(cluster='${PROJECT}-cluster',services=['${PROJECT}-api']); print(r['services'][0]['runningCount'], 'running')\""
