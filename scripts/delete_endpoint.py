"""
delete_endpoint.py
------------------
Deletes the BGE-M3 SageMaker endpoint to stop billing after demo.
Run this when you're done demoing.

Usage:
    python scripts/delete_endpoint.py
"""

import boto3

REGION        = "ap-south-1"
ENDPOINT_NAME = "bge-m3-krishirakshak"

sm = boto3.client("sagemaker", region_name=REGION)

try:
    sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
    print(f"Endpoint '{ENDPOINT_NAME}' deleted. Billing stopped.")
except sm.exceptions.ClientError as e:
    print(f"Could not delete endpoint: {e}")
