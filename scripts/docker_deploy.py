"""
Full ECR build + push + ECS redeploy via Python Docker SDK.
No AWS CLI or bash piping needed.
Usage: python scripts/docker_deploy.py
"""
import base64
import subprocess
import sys
import boto3
import docker

REGION   = "us-east-1"
ACCOUNT  = "593755927741"
REGISTRY = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com"
REPO     = f"{REGISTRY}/krishirakshak-api"
PROJECT  = "krishirakshak"
API_URL  = "https://sgy86f7n6l.execute-api.us-east-1.amazonaws.com"

def get_git_tag():
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True)
        return result.stdout.strip() or "latest"
    except Exception:
        return "latest"

def ecr_login(client):
    print("[1/4] Logging in to ECR...")
    t   = boto3.client("ecr", region_name=REGION).get_authorization_token()
    raw = base64.b64decode(t["authorizationData"][0]["authorizationToken"]).decode()
    user, pwd = raw.split(":", 1)
    result = client.login(username=user, password=pwd, registry=REGISTRY)
    print(f"      {result.get('Status', 'OK')}")

def build_image(client, tag):
    print(f"[2/4] Building Docker image (tag={tag})...")
    print("      This takes 5-10 min on first build...")
    image, logs = client.images.build(
        path=".",
        dockerfile="Dockerfile",
        tag=f"{REPO}:{tag}",
        rm=True,
    )
    for chunk in logs:
        if "stream" in chunk:
            line = chunk["stream"].strip()
            if line:
                print(f"      {line}")
    # also tag as latest
    image.tag(REPO, "latest")
    print(f"      Build complete: {image.short_id}")
    return image

def push_image(client, tag):
    print(f"[3/4] Pushing to ECR...")
    for t in [tag, "latest"]:
        print(f"      Pushing {REPO}:{t} ...")
        for chunk in client.images.push(REPO, tag=t, stream=True, decode=True):
            if "error" in chunk:
                print(f"      ERROR: {chunk['error']}")
                sys.exit(1)
            if "status" in chunk and chunk.get("progressDetail"):
                pass  # skip noisy progress lines
            elif "status" in chunk:
                print(f"      {chunk['status']}")
    print("      Push complete")

def redeploy_ecs():
    print("[4/4] Triggering ECS redeployment...")
    boto3.client("ecs", region_name=REGION).update_service(
        cluster=f"{PROJECT}-cluster",
        service=f"{PROJECT}-api",
        forceNewDeployment=True,
    )
    print("      ECS redeployment triggered")

def main():
    tag    = get_git_tag()
    client = docker.from_env(timeout=600)

    ecr_login(client)
    build_image(client, tag)
    push_image(client, tag)
    redeploy_ecs()

    print(f"\n=== Deploy complete ===")
    print(f"API URL    : {API_URL}")
    print(f"Health     : {API_URL}/v1/health")
    print(f"Image tag  : {tag}")
    print(f"\nECS takes ~2 min to pull and start the new task.")

if __name__ == "__main__":
    main()
