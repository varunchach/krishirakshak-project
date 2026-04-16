"""
Provision AWS infrastructure for KrishiRakshak (replaces Terraform).

Creates:
  - CloudWatch log groups
  - DynamoDB table
  - ECR repository
  - IAM roles (ECS task + execution)
  - Security groups (API GW VPC link, ALB, ECS)
  - Internal ALB + target group + listener
  - ECS cluster + task definition + service
  - API Gateway VPC link + HTTP API + routes + stage

Usage:
    python scripts/provision.py [--env dev] [--region us-east-1] [--account 593755927741]
"""

import argparse
import json
import time
import boto3
from botocore.exceptions import ClientError

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT        = "krishirakshak"
REGION         = "us-east-1"
SM_REGION      = "ap-south-1"
ACCOUNT        = "593755927741"
VPC_ID         = "vpc-0e0318f7665ce160e"
SUBNET_IDS     = [
    "subnet-0f0b03dad65cdb844",  # us-east-1d use1-az2
    "subnet-0ac60f15c5b2310ae",  # us-east-1a use1-az4
    "subnet-00bb4338053a9dece",  # us-east-1f use1-az5
    "subnet-0372bd0af3868d46f",  # us-east-1b use1-az6
    "subnet-0504ddee11f65813d",  # us-east-1c use1-az1
    # subnet-0baef118b484d6f99 (use1-az3) excluded — not supported for API GW VPC links
]

TAGS      = [{"Key": "Project", "Value": PROJECT}, {"Key": "ManagedBy", "Value": "provision.py"}]
TAGS_ECS  = [{"key": "Project", "value": PROJECT}, {"key": "ManagedBy", "value": "provision.py"}]


def tag_dict(tags): return {t["Key"]: t["Value"] for t in tags}


def exists(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in (
            "ResourceNotFoundException", "NoSuchEntity",
            "RepositoryNotFoundException", "ValidationException",
            "ClusterNotFoundException",
        ):
            return False
        raise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env",     default="dev")
    p.add_argument("--region",  default=REGION)
    p.add_argument("--account", default=ACCOUNT)
    return p.parse_args()


# ── 1. CloudWatch log groups ───────────────────────────────────────────────────
def provision_log_groups(logs, project, env):
    for name in [f"/{project}/api", f"/{project}/api-gateway", f"/{project}/ui"]:
        try:
            logs.create_log_group(logGroupName=name, tags=tag_dict(TAGS))
            logs.put_retention_policy(logGroupName=name, retentionInDays=30)
            print(f"  Created log group: {name}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
                print(f"  Log group exists: {name}")
            else:
                raise


# ── 2. DynamoDB ────────────────────────────────────────────────────────────────
def provision_dynamodb(ddb, project, env):
    name = f"{project}-predictions-{env}"
    try:
        ddb.create_table(
            TableName=name,
            BillingMode="PAY_PER_REQUEST",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            Tags=TAGS,
        )
        ddb.get_waiter("table_exists").wait(TableName=name)
        ddb.update_continuous_backups(
            TableName=name,
            PointInTimeRecoverySpecification={"PointInTimeRecoveryEnabled": True},
        )
        print(f"  Created DynamoDB table: {name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print(f"  DynamoDB table exists: {name}")
        else:
            raise
    return name


# ── 3. ECR ─────────────────────────────────────────────────────────────────────
def provision_ecr(ecr, project):
    name = f"{project}-api"
    try:
        resp = ecr.create_repository(
            repositoryName=name,
            imageScanningConfiguration={"scanOnPush": True},
            imageTagMutability="MUTABLE",
            tags=TAGS,
        )
        url = resp["repository"]["repositoryUri"]
        print(f"  Created ECR repo: {url}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryAlreadyExistsException":
            url = ecr.describe_repositories(repositoryNames=[name])["repositories"][0]["repositoryUri"]
            print(f"  ECR repo exists: {url}")
        else:
            raise
    return url


# ── 4. IAM roles ───────────────────────────────────────────────────────────────
def _create_role(iam, name, service, description):
    try:
        iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=json.dumps({
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}],
            }),
            Description=description,
            Tags=TAGS,
        )
        print(f"  Created IAM role: {name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            print(f"  IAM role exists: {name}")
        else:
            raise


def provision_iam(iam, project, env, account, assets_bucket_arn, dynamodb_arn, log_group_arns):
    exec_role = f"{project}-ecs-execution-role"
    task_role = f"{project}-ecs-task-role"

    _create_role(iam, exec_role, "ecs-tasks.amazonaws.com", "ECS task execution role")
    iam.attach_role_policy(
        RoleName=exec_role,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
    )

    _create_role(iam, task_role, "ecs-tasks.amazonaws.com", "ECS task role")
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["sagemaker:InvokeEndpoint", "sagemaker:DescribeEndpoint"],
             "Resource": [
                 f"arn:aws:sagemaker:{SM_REGION}:{account}:endpoint/bge-m3-krishirakshak",
                 f"arn:aws:sagemaker:{SM_REGION}:{account}:endpoint/krishirakshak-efficientnet-b3",
             ]},
            {"Effect": "Allow", "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"], "Resource": "*"},
            {"Effect": "Allow", "Action": ["polly:SynthesizeSpeech"], "Resource": "*"},
            {"Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
             "Resource": [assets_bucket_arn, f"{assets_bucket_arn}/*"]},
            {"Effect": "Allow", "Action": ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query", "dynamodb:UpdateItem"],
             "Resource": dynamodb_arn},
            {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
             "Resource": [f"{arn}:*" for arn in log_group_arns]},
            {"Effect": "Allow", "Action": ["cloudwatch:PutMetricData"], "Resource": "*"},
        ],
    }
    try:
        iam.put_role_policy(RoleName=task_role, PolicyName=f"{project}-ecs-task-policy", PolicyDocument=json.dumps(policy))
        print(f"  Updated IAM task policy")
    except ClientError:
        raise

    exec_arn = f"arn:aws:iam::{account}:role/{exec_role}"
    task_arn = f"arn:aws:iam::{account}:role/{task_role}"
    return exec_arn, task_arn


# ── 5. Security groups ─────────────────────────────────────────────────────────
def _sg_id(ec2, name):
    resp = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [name]}, {"Name": "vpc-id", "Values": [VPC_ID]}])
    groups = resp["SecurityGroups"]
    return groups[0]["GroupId"] if groups else None


def provision_security_groups(ec2, project):
    def create_sg(name, desc):
        sg_id = _sg_id(ec2, name)
        if sg_id:
            print(f"  SG exists: {name} ({sg_id})")
            return sg_id
        resp = ec2.create_security_group(GroupName=name, Description=desc, VpcId=VPC_ID, TagSpecifications=[
            {"ResourceType": "security-group", "Tags": TAGS}
        ])
        sg_id = resp["GroupId"]
        print(f"  Created SG: {name} ({sg_id})")
        return sg_id

    apigw_sg = create_sg(f"{project}-apigw-vpc-link-sg", "API Gateway VPC link SG")
    alb_sg   = create_sg(f"{project}-alb-sg", "ALB SG")
    api_sg   = create_sg(f"{project}-api-sg", "ECS task SG")

    # Authorize ingress rules (ignore if already exists)
    def auth(sg_id, from_port, to_port, protocol, source_sg=None, cidr=None):
        try:
            kwargs = {"GroupId": sg_id, "IpPermissions": [{"IpProtocol": protocol, "FromPort": from_port, "ToPort": to_port}]}
            if source_sg:
                kwargs["IpPermissions"][0]["UserIdGroupPairs"] = [{"GroupId": source_sg}]
            elif cidr:
                kwargs["IpPermissions"][0]["IpRanges"] = [{"CidrIp": cidr}]
            ec2.authorize_security_group_ingress(**kwargs)
        except ClientError as e:
            if e.response["Error"]["Code"] != "InvalidPermission.Duplicate":
                raise

    auth(alb_sg,  80,   80,   "tcp", source_sg=apigw_sg)
    auth(api_sg,  8000, 8000, "tcp", source_sg=alb_sg)

    return apigw_sg, alb_sg, api_sg


# ── 6. ALB ─────────────────────────────────────────────────────────────────────
def provision_alb(elb, project, env, alb_sg):
    alb_name = f"{project}-alb"
    tg_name  = f"{project}-tg"

    # Target group
    try:
        tg = elb.describe_target_groups(Names=[tg_name])["TargetGroups"][0]
        tg_arn = tg["TargetGroupArn"]
        print(f"  Target group exists: {tg_name}")
    except ClientError:
        tg_arn = elb.create_target_group(
            Name=tg_name, Protocol="HTTP", Port=8000,
            VpcId=VPC_ID, TargetType="ip",
            HealthCheckPath="/v1/health", HealthCheckIntervalSeconds=30,
            HealthCheckTimeoutSeconds=10, HealthyThresholdCount=2,
            UnhealthyThresholdCount=3, Matcher={"HttpCode": "200"},
            Tags=TAGS,
        )["TargetGroups"][0]["TargetGroupArn"]
        print(f"  Created target group: {tg_name}")

    # ALB
    try:
        alb = elb.describe_load_balancers(Names=[alb_name])["LoadBalancers"][0]
        alb_arn = alb["LoadBalancerArn"]
        alb_dns = alb["DNSName"]
        print(f"  ALB exists: {alb_name}")
    except ClientError:
        alb = elb.create_load_balancer(
            Name=alb_name, Scheme="internal", Type="application",
            SecurityGroups=[alb_sg], Subnets=SUBNET_IDS, Tags=TAGS,
        )["LoadBalancers"][0]
        alb_arn = alb["LoadBalancerArn"]
        alb_dns = alb["DNSName"]
        elb.get_waiter("load_balancer_available").wait(LoadBalancerArns=[alb_arn])
        print(f"  Created ALB: {alb_name}")

    # Listener
    listeners = elb.describe_listeners(LoadBalancerArn=alb_arn)["Listeners"]
    if not listeners:
        listener = elb.create_listener(
            LoadBalancerArn=alb_arn, Protocol="HTTP", Port=80,
            DefaultActions=[{"Type": "forward", "TargetGroupArn": tg_arn}],
        )["Listeners"][0]
        listener_arn = listener["ListenerArn"]
        print(f"  Created ALB listener")
    else:
        listener_arn = listeners[0]["ListenerArn"]
        print(f"  ALB listener exists")

    return alb_arn, alb_dns, tg_arn, listener_arn


# ── 7. ECS cluster ─────────────────────────────────────────────────────────────
def provision_ecs_cluster(ecs, project):
    name = f"{project}-cluster"
    clusters = ecs.describe_clusters(clusters=[name])["clusters"]
    if clusters and clusters[0]["status"] == "ACTIVE":
        print(f"  ECS cluster exists: {name}")
    else:
        ecs.create_cluster(clusterName=name, settings=[{"name": "containerInsights", "value": "enabled"}], tags=TAGS_ECS)
        print(f"  Created ECS cluster: {name}")
    return name


# ── 8. ECS task definition + service ──────────────────────────────────────────
def provision_ecs_service(ecs, project, env, ecr_url, exec_role_arn, task_role_arn, cluster_name, tg_arn, api_sg):
    container = {
        "name": "api",
        "image": f"{ecr_url}:latest",
        "essential": True,
        "portMappings": [{"containerPort": 8000, "hostPort": 8000, "protocol": "tcp"}],
        "environment": [
            {"name": "AWS_DEFAULT_REGION",            "value": REGION},
            {"name": "S3_BUCKET",                     "value": f"{project}-assets-{env}"},
            {"name": "FAISS_S3_BUCKET",               "value": f"{project}-assets-{env}"},
            {"name": "FAISS_S3_KEY",                  "value": "faiss_index/store.pkl"},
            {"name": "DYNAMODB_TABLE",                "value": f"{project}-predictions-{env}"},
            {"name": "SAGEMAKER_REGION",              "value": SM_REGION},
            {"name": "SAGEMAKER_ENDPOINT",            "value": "bge-m3-krishirakshak"},
            {"name": "CLASSIFIER_BACKEND",            "value": "sagemaker"},
            {"name": "CLASSIFIER_SAGEMAKER_REGION",   "value": SM_REGION},
            {"name": "CLASSIFIER_SAGEMAKER_ENDPOINT", "value": "krishirakshak-efficientnet-b3"},
            {"name": "WORKERS",                       "value": "2"},
        ],
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group":         f"/{project}/api",
                "awslogs-region":        REGION,
                "awslogs-stream-prefix": "api",
            },
        },
        "healthCheck": {
            "command":     ["CMD-SHELL", "curl -f http://localhost:8000/v1/health || exit 1"],
            "interval":    30, "timeout": 10, "retries": 3, "startPeriod": 90,
        },
    }

    td = ecs.register_task_definition(
        family=f"{project}-api",
        requiresCompatibilities=["FARGATE"],
        networkMode="awsvpc",
        cpu="1024", memory="2048",
        executionRoleArn=exec_role_arn,
        taskRoleArn=task_role_arn,
        containerDefinitions=[container],
        tags=TAGS_ECS,
    )
    td_arn = td["taskDefinition"]["taskDefinitionArn"]
    print(f"  Registered task definition: {project}-api")

    svc_name = f"{project}-api"
    services = ecs.describe_services(cluster=cluster_name, services=[svc_name])["services"]
    if services and services[0]["status"] == "ACTIVE":
        ecs.update_service(
            cluster=cluster_name, service=svc_name,
            taskDefinition=td_arn, desiredCount=1,
        )
        print(f"  Updated ECS service: {svc_name}")
    else:
        ecs.create_service(
            cluster=cluster_name, serviceName=svc_name,
            taskDefinition=td_arn, desiredCount=1,
            launchType="FARGATE",
            networkConfiguration={"awsvpcConfiguration": {
                "subnets": SUBNET_IDS,
                "securityGroups": [api_sg],
                "assignPublicIp": "ENABLED",
            }},
            loadBalancers=[{"targetGroupArn": tg_arn, "containerName": "api", "containerPort": 8000}],
            tags=TAGS_ECS,
        )
        print(f"  Created ECS service: {svc_name}")


# ── 9. Streamlit public ALB + ECS service ─────────────────────────────────────
def provision_streamlit(ecr, elb, ecs, ec2, project, env, exec_role_arn, task_role_arn, cluster_name, api_url):
    # ECR repo for Streamlit image
    ui_repo = f"{project}-ui"
    try:
        resp    = ecr.create_repository(repositoryName=ui_repo, imageScanningConfiguration={"scanOnPush": True}, tags=TAGS)
        ui_url  = resp["repository"]["repositoryUri"]
        print(f"  Created ECR repo: {ui_url}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryAlreadyExistsException":
            ui_url = ecr.describe_repositories(repositoryNames=[ui_repo])["repositories"][0]["repositoryUri"]
            print(f"  ECR repo exists: {ui_url}")
        else:
            raise

    # Security group — allow port 8501 from internet
    ui_sg_name = f"{project}-ui-sg"
    ui_sg = _sg_id(ec2, ui_sg_name)
    if not ui_sg:
        resp  = ec2.create_security_group(GroupName=ui_sg_name, Description="Streamlit UI SG", VpcId=VPC_ID,
                                          TagSpecifications=[{"ResourceType": "security-group", "Tags": TAGS}])
        ui_sg = resp["GroupId"]
        ec2.authorize_security_group_ingress(GroupId=ui_sg, IpPermissions=[
            {"IpProtocol": "tcp", "FromPort": 8501, "ToPort": 8501, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
        ])
        print(f"  Created UI SG: {ui_sg}")
    else:
        print(f"  UI SG exists: {ui_sg}")

    # Public ALB for Streamlit
    ui_alb_name = f"{project}-ui-alb"
    ui_tg_name  = f"{project}-ui-tg"

    try:
        tg      = elb.describe_target_groups(Names=[ui_tg_name])["TargetGroups"][0]
        ui_tg   = tg["TargetGroupArn"]
        print(f"  UI target group exists")
    except ClientError:
        ui_tg = elb.create_target_group(
            Name=ui_tg_name, Protocol="HTTP", Port=8501, VpcId=VPC_ID, TargetType="ip",
            HealthCheckPath="/_stcore/health", HealthCheckIntervalSeconds=30,
            HealthCheckTimeoutSeconds=10, HealthyThresholdCount=2, UnhealthyThresholdCount=3,
            Matcher={"HttpCode": "200"}, Tags=TAGS,
        )["TargetGroups"][0]["TargetGroupArn"]
        print(f"  Created UI target group")

    try:
        alb     = elb.describe_load_balancers(Names=[ui_alb_name])["LoadBalancers"][0]
        ui_alb_dns = alb["DNSName"]
        ui_alb_arn = alb["LoadBalancerArn"]
        print(f"  UI ALB exists: {ui_alb_dns}")
    except ClientError:
        alb        = elb.create_load_balancer(
            Name=ui_alb_name, Scheme="internet-facing", Type="application",
            SecurityGroups=[ui_sg], Subnets=SUBNET_IDS, Tags=TAGS,
        )["LoadBalancers"][0]
        ui_alb_arn = alb["LoadBalancerArn"]
        ui_alb_dns = alb["DNSName"]
        elb.get_waiter("load_balancer_available").wait(LoadBalancerArns=[ui_alb_arn])
        print(f"  Created UI ALB: {ui_alb_dns}")

    listeners = elb.describe_listeners(LoadBalancerArn=ui_alb_arn)["Listeners"]
    if not listeners:
        elb.create_listener(
            LoadBalancerArn=ui_alb_arn, Protocol="HTTP", Port=80,
            DefaultActions=[{"Type": "forward", "TargetGroupArn": ui_tg}],
        )
        print(f"  Created UI ALB listener")

    # ECS task definition for Streamlit
    container = {
        "name"        : "ui",
        "image"       : f"{ui_url}:latest",
        "essential"   : True,
        "portMappings": [{"containerPort": 8501, "hostPort": 8501, "protocol": "tcp"}],
        "environment" : [{"name": "API_URL", "value": api_url}],
        "logConfiguration": {
            "logDriver": "awslogs",
            "options"  : {
                "awslogs-group"        : f"/{project}/ui",
                "awslogs-region"       : REGION,
                "awslogs-stream-prefix": "ui",
            },
        },
        "healthCheck": {
            "command"    : ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"],
            "interval"   : 30, "timeout": 10, "retries": 3, "startPeriod": 60,
        },
    }

    td     = ecs.register_task_definition(
        family=f"{project}-ui", requiresCompatibilities=["FARGATE"],
        networkMode="awsvpc", cpu="512", memory="1024",
        executionRoleArn=exec_role_arn, taskRoleArn=task_role_arn,
        containerDefinitions=[container], tags=TAGS_ECS,
    )
    td_arn = td["taskDefinition"]["taskDefinitionArn"]
    print(f"  Registered UI task definition")

    svc_name = f"{project}-ui"
    services = ecs.describe_services(cluster=cluster_name, services=[svc_name])["services"]
    if services and services[0]["status"] == "ACTIVE":
        ecs.update_service(cluster=cluster_name, service=svc_name, taskDefinition=td_arn, desiredCount=1)
        print(f"  Updated UI ECS service")
    else:
        ecs.create_service(
            cluster=cluster_name, serviceName=svc_name,
            taskDefinition=td_arn, desiredCount=1, launchType="FARGATE",
            networkConfiguration={"awsvpcConfiguration": {
                "subnets": SUBNET_IDS, "securityGroups": [ui_sg], "assignPublicIp": "ENABLED",
            }},
            loadBalancers=[{"targetGroupArn": ui_tg, "containerName": "ui", "containerPort": 8501}],
            tags=TAGS_ECS,
        )
        print(f"  Created UI ECS service")

    return ui_url, ui_alb_dns


# ── 10. API Gateway ─────────────────────────────────────────────────────────────
def provision_apigw(apigw, project, apigw_sg, listener_arn):
    # VPC link
    links = apigw.get_vpc_links()["Items"]
    link = next((l for l in links if l["Name"] == f"{project}-vpc-link"), None)
    if link:
        link_id = link["VpcLinkId"]
        print(f"  VPC link exists: {link_id}")
    else:
        link = apigw.create_vpc_link(
            Name=f"{project}-vpc-link",
            SubnetIds=SUBNET_IDS,
            SecurityGroupIds=[apigw_sg],
            Tags=tag_dict(TAGS),
        )
        link_id = link["VpcLinkId"]
        print(f"  Created VPC link: {link_id} (waiting ~2 min for AVAILABLE...)")
        for _ in range(24):
            status = apigw.get_vpc_link(VpcLinkId=link_id)["VpcLinkStatus"]
            if status == "AVAILABLE":
                break
            time.sleep(5)
        print(f"  VPC link ready")

    # API
    apis = apigw.get_apis()["Items"]
    api = next((a for a in apis if a["Name"] == f"{project}-http-api"), None)
    if api:
        api_id = api["ApiId"]
        print(f"  API Gateway exists: {api_id}")
    else:
        api_id = apigw.create_api(
            Name=f"{project}-http-api", ProtocolType="HTTP",
            Tags=tag_dict(TAGS),
        )["ApiId"]
        print(f"  Created API Gateway: {api_id}")

    # Integration
    integrations = apigw.get_integrations(ApiId=api_id)["Items"]
    if not integrations:
        integration_id = apigw.create_integration(
            ApiId=api_id,
            IntegrationType="HTTP_PROXY",
            IntegrationMethod="ANY",
            IntegrationUri=listener_arn,
            ConnectionType="VPC_LINK",
            ConnectionId=link_id,
            PayloadFormatVersion="1.0",
            TimeoutInMillis=30000,
        )["IntegrationId"]
        print(f"  Created API Gateway integration")

        apigw.create_route(ApiId=api_id, RouteKey="GET /v1/health",
                           Target=f"integrations/{integration_id}", AuthorizationType="NONE")
        apigw.create_route(ApiId=api_id, RouteKey="ANY /v1/{proxy+}",
                           Target=f"integrations/{integration_id}", AuthorizationType="NONE")
        print(f"  Created routes")
    else:
        integration_id = integrations[0]["IntegrationId"]
        print(f"  Integration exists")

    # Stage
    try:
        apigw.get_stage(ApiId=api_id, StageName="$default")
        print(f"  Stage exists")
    except ClientError:
        apigw.create_stage(ApiId=api_id, StageName="$default", AutoDeploy=True, Tags=tag_dict(TAGS))
        print(f"  Created stage")

    invoke_url = f"https://{api_id}.execute-api.{REGION}.amazonaws.com"
    return invoke_url


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args    = parse_args()
    project = PROJECT
    env     = args.env
    region  = args.region
    account = args.account

    print(f"\n=== Provisioning {project}-{env} in {region} ===\n")

    logs  = boto3.client("logs",           region_name=region)
    ddb   = boto3.client("dynamodb",       region_name=region)
    ecr   = boto3.client("ecr",            region_name=region)
    iam   = boto3.client("iam")
    ec2   = boto3.client("ec2",            region_name=region)
    elb   = boto3.client("elbv2",          region_name=region)
    ecs   = boto3.client("ecs",            region_name=region)
    apigw = boto3.client("apigatewayv2",   region_name=region)

    print("[1/8] CloudWatch log groups")
    provision_log_groups(logs, project, env)

    print("[2/8] DynamoDB")
    ddb_name = provision_dynamodb(ddb, project, env)
    ddb_arn  = f"arn:aws:dynamodb:{region}:{account}:table/{ddb_name}"

    print("[3/8] ECR")
    ecr_url = provision_ecr(ecr, project)

    print("[4/8] Security groups")
    apigw_sg, alb_sg, api_sg = provision_security_groups(ec2, project)

    print("[5/8] ALB")
    alb_arn, alb_dns, tg_arn, listener_arn = provision_alb(elb, project, env, alb_sg)

    print("[6/8] IAM roles")
    assets_arn = f"arn:aws:s3:::{project}-assets-{env}"
    log_arns   = [f"arn:aws:logs:{region}:{account}:log-group:/{project}/api",
                  f"arn:aws:logs:{region}:{account}:log-group:/{project}/api-gateway"]
    exec_arn, task_arn = provision_iam(iam, project, env, account, assets_arn, ddb_arn, log_arns)

    print("[7/8] ECS cluster + service")
    cluster = provision_ecs_cluster(ecs, project)
    provision_ecs_service(ecs, project, env, ecr_url, exec_arn, task_arn, cluster, tg_arn, api_sg)

    print("[8/9] API Gateway")
    invoke_url = provision_apigw(apigw, project, apigw_sg, listener_arn)

    print("[9/9] Streamlit UI")
    ui_ecr_url, ui_alb_dns = provision_streamlit(
        ecr, elb, ecs, ec2, project, env,
        exec_arn, task_arn, cluster,
        api_url=f"{invoke_url}/v1",
    )

    print(f"""
=== Done ===
ECR API repo : {ecr_url}
ECR UI repo  : {ui_ecr_url}
DynamoDB     : {ddb_name}
API URL      : {invoke_url}
UI URL       : http://{ui_alb_dns}

Next: .\\scripts\\deploy.ps1
""")


if __name__ == "__main__":
    main()
