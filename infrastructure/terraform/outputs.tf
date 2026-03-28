output "s3_images_bucket" {
  value = aws_s3_bucket.images.id
}

output "s3_models_bucket" {
  value = aws_s3_bucket.models.id
}

output "dynamodb_table" {
  value = aws_dynamodb_table.predictions.name
}

output "ecr_repo_url" {
  value = aws_ecr_repository.api.repository_url
}

output "sns_topic_arn" {
  value = aws_sns_topic.alerts.arn
}
