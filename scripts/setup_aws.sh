#!/bin/bash
# One-time AWS setup for KrishiRakshak
set -euo pipefail

PROJECT="krishirakshak"
REGION="us-east-1"

echo "=== KrishiRakshak AWS Setup ==="

# 1. Create SageMaker execution role
echo "Creating SageMaker execution role..."
aws iam create-role \
  --role-name SageMakerExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }' 2>/dev/null || echo "Role already exists"

aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess 2>/dev/null || true

aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess 2>/dev/null || true

# 2. Enable Bedrock model access
echo ""
echo "MANUAL STEP: Enable Bedrock model access"
echo "  Go to: https://console.aws.amazon.com/bedrock/home#/modelaccess"
echo "  Enable: Claude 3 Sonnet (or Amazon Titan Text)"
echo ""

# 3. Store Sarvam API key in SSM Parameter Store (free, encrypted)
echo "Enter your Sarvam AI API key (from https://www.sarvam.ai/):"
read -s SARVAM_KEY
aws ssm put-parameter \
  --name "/$PROJECT/sarvam-api-key" \
  --value "$SARVAM_KEY" \
  --type "SecureString" \
  --overwrite \
  --region "$REGION"
echo "Sarvam API key stored in SSM Parameter Store"

# 4. Create S3 buckets
echo "Creating S3 buckets..."
aws s3 mb "s3://$PROJECT-images-dev" --region "$REGION" 2>/dev/null || echo "Bucket exists"
aws s3 mb "s3://$PROJECT-models-dev" --region "$REGION" 2>/dev/null || echo "Bucket exists"

# 5. Set billing alarm
echo "Setting billing alarm at \$50..."
aws cloudwatch put-metric-alarm \
  --alarm-name "$PROJECT-billing-50" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --threshold 50 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --dimensions "Name=Currency,Value=USD" \
  --region us-east-1 2>/dev/null || echo "Alarm exists"

echo ""
echo "=== Setup complete! ==="
echo "Next: bash scripts/download_dataset.sh"
