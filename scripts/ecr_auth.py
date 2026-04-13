"""Write ECR auth token directly to ~/.docker/config.json"""
import boto3, base64, json, os, sys

print("step1: imports ok", flush=True)

REGION   = "us-east-1"
ACCOUNT  = "593755927741"
REGISTRY = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com"

print("step2: calling ECR...", flush=True)
t     = boto3.client("ecr", region_name=REGION).get_authorization_token()
token = t["authorizationData"][0]["authorizationToken"]
print("step3: got token", flush=True)

config_path = os.path.join(os.path.expanduser("~"), ".docker", "config.json")
os.makedirs(os.path.dirname(config_path), exist_ok=True)
print(f"step4: config path = {config_path}", flush=True)

try:
    with open(config_path) as f:
        config = json.load(f)
except Exception:
    config = {}

config.setdefault("auths", {})[REGISTRY] = {"auth": token}
config.pop("credsStore", None)

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Done. ECR credentials written for {REGISTRY}", flush=True)
