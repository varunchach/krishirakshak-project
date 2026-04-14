"""
create_realtime_endpoint.py
---------------------------
Switches BGE-M3 from serverless → real-time GPU endpoint (ml.g4dn.xlarge).
No cold starts. Inference ~200ms vs 1-3s on serverless.

Usage:
    python scripts/create_realtime_endpoint.py

Cost: ~$0.53/hr — run `python scripts/delete_endpoint.py` after demo to stop billing.
"""

import boto3
import time

REGION        = "ap-south-1"
ENDPOINT_NAME = "bge-m3-krishirakshak"
MODEL_NAME    = "bge-m3-krishirakshak-model"
INSTANCE_TYPE = "ml.g4dn.xlarge"

sm = boto3.client("sagemaker", region_name=REGION)

# ── Check if endpoint already exists ──────────────────────────────────────────
try:
    desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    current_type = desc.get("ProductionVariants", [{}])[0].get("CurrentInstanceType", "serverless")
    print(f"Endpoint '{ENDPOINT_NAME}' already exists ({current_type})")
    print("Delete it first or rename ENDPOINT_NAME to create a new one.")
    exit(0)
except sm.exceptions.ClientError:
    pass  # does not exist yet, proceed

# ── Get existing model name from current endpoint config ──────────────────────
try:
    configs = sm.list_endpoint_configs(NameContains="bge-m3")["EndpointConfigs"]
    existing_config = configs[0]["EndpointConfigName"] if configs else None
    if existing_config:
        cfg = sm.describe_endpoint_config(EndpointConfigName=existing_config)
        MODEL_NAME = cfg["ProductionVariants"][0]["ModelName"]
        print(f"Reusing existing model: {MODEL_NAME}")
except Exception as e:
    print(f"Could not find existing model config: {e}")
    print("Make sure BGE-M3 model is already registered in SageMaker.")
    exit(1)

# ── Create real-time endpoint config ──────────────────────────────────────────
config_name = f"{ENDPOINT_NAME}-realtime-config"
print(f"Creating endpoint config: {config_name}")
sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        "VariantName"   : "primary",
        "ModelName"     : MODEL_NAME,
        "InstanceType"  : INSTANCE_TYPE,
        "InitialInstanceCount": 1,
    }],
)

# ── Create endpoint ────────────────────────────────────────────────────────────
print(f"Creating real-time endpoint: {ENDPOINT_NAME} on {INSTANCE_TYPE}")
print("This takes ~5-8 minutes...")
sm.create_endpoint(
    EndpointName      =ENDPOINT_NAME,
    EndpointConfigName=config_name,
)

# ── Wait for InService ─────────────────────────────────────────────────────────
for i in range(60):
    status = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
    print(f"  [{i*10}s] Status: {status}")
    if status == "InService":
        print(f"\nEndpoint '{ENDPOINT_NAME}' is InService on {INSTANCE_TYPE}")
        print(f"Cost: ~$0.53/hr — remember to delete after demo!")
        print(f"  python scripts/delete_endpoint.py")
        break
    elif status == "Failed":
        print("Endpoint creation FAILED. Check SageMaker console.")
        break
    time.sleep(10)
