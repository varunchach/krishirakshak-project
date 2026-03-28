#!/bin/bash
# Tear down all AWS resources to save credits
set -euo pipefail

PROJECT="krishirakshak"

echo "=== Cleaning up KrishiRakshak resources ==="
echo "WARNING: This will delete all resources. Press Ctrl+C to cancel."
sleep 5

# SageMaker
echo "Deleting SageMaker endpoint..."
aws sagemaker delete-endpoint --endpoint-name "$PROJECT-florence2" 2>/dev/null || true
aws sagemaker delete-endpoint-config --endpoint-config-name "$PROJECT-florence2-config" 2>/dev/null || true
aws sagemaker delete-model --model-name "$PROJECT-florence2-model" 2>/dev/null || true

# S3 (empty then delete)
echo "Emptying S3 buckets..."
aws s3 rm "s3://$PROJECT-images-dev" --recursive 2>/dev/null || true
aws s3 rm "s3://$PROJECT-models-dev" --recursive 2>/dev/null || true
aws s3 rb "s3://$PROJECT-images-dev" 2>/dev/null || true
aws s3 rb "s3://$PROJECT-models-dev" 2>/dev/null || true

# DynamoDB
echo "Deleting DynamoDB table..."
aws dynamodb delete-table --table-name "$PROJECT-predictions" 2>/dev/null || true

# Secrets
echo "Deleting secrets..."
aws secretsmanager delete-secret --secret-id "$PROJECT/sarvam-api-key" --force-delete-without-recovery 2>/dev/null || true

# Terraform (if used)
if [ -d "infrastructure/terraform/.terraform" ]; then
    echo "Running terraform destroy..."
    cd infrastructure/terraform
    terraform destroy -auto-approve
    cd ../..
fi

echo ""
echo "=== Cleanup complete! ==="
echo "Sarvam AI credits are NOT affected (they persist forever)."
