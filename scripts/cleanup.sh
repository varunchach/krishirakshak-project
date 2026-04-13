#!/bin/bash
# Tear down all KrishiRakshak AWS resources to stop billing.
# Usage: bash scripts/cleanup.sh
set -euo pipefail

PROJECT="krishirakshak"
REGION="us-east-1"
SM_REGION="ap-south-1"

echo "=== KrishiRakshak Cleanup ==="
echo "WARNING: This will delete all resources. Press Ctrl+C to cancel."
sleep 5

python -c "
import boto3, time

project   = '${PROJECT}'
region    = '${REGION}'
sm_region = '${SM_REGION}'

ecs   = boto3.client('ecs',          region_name=region)
elb   = boto3.client('elbv2',        region_name=region)
ec2   = boto3.client('ec2',          region_name=region)
apigw = boto3.client('apigatewayv2', region_name=region)
ecr   = boto3.client('ecr',          region_name=region)
ddb   = boto3.client('dynamodb',     region_name=region)
s3    = boto3.client('s3',           region_name=region)
logs  = boto3.client('logs',         region_name=region)

def safe(fn, *a, **kw):
    try: fn(*a, **kw)
    except Exception as e: print(f'  skip: {e}')

print('ECS...')
safe(ecs.update_service, cluster=f'{project}-cluster', service=f'{project}-api', desiredCount=0)
time.sleep(5)
safe(ecs.delete_service, cluster=f'{project}-cluster', service=f'{project}-api', force=True)
safe(ecs.delete_cluster, cluster=f'{project}-cluster')

print('API Gateway...')
for api in apigw.get_apis().get('Items', []):
    if project in api['Name']:
        safe(apigw.delete_api, ApiId=api['ApiId'])
for link in apigw.get_vpc_links().get('Items', []):
    if project in link['Name']:
        safe(apigw.delete_vpc_link, VpcLinkId=link['VpcLinkId'])

print('ALB...')
for lb in elb.describe_load_balancers().get('LoadBalancers', []):
    if project in lb['LoadBalancerName']:
        for tg in elb.describe_target_groups(LoadBalancerArn=lb['LoadBalancerArn']).get('TargetGroups', []):
            safe(elb.delete_target_group, TargetGroupArn=tg['TargetGroupArn'])
        safe(elb.delete_load_balancer, LoadBalancerArn=lb['LoadBalancerArn'])

print('Security groups...')
for sg in ec2.describe_security_groups(Filters=[{'Name':'tag:Project','Values':[project]}]).get('SecurityGroups', []):
    safe(ec2.delete_security_group, GroupId=sg['GroupId'])

print('ECR...')
safe(ecr.delete_repository, repositoryName=f'{project}-api', force=True)

print('DynamoDB...')
safe(ddb.delete_table, TableName=f'{project}-predictions-dev')

print('S3...')
try:
    objs = s3.list_objects_v2(Bucket=f'{project}-assets-dev').get('Contents', [])
    if objs:
        s3.delete_objects(Bucket=f'{project}-assets-dev', Delete={'Objects': [{'Key': o['Key']} for o in objs]})
    s3.delete_bucket(Bucket=f'{project}-assets-dev')
except Exception as e:
    print(f'  S3: {e}')

print('Log groups...')
for name in [f'/{project}/api', f'/{project}/api-gateway']:
    safe(logs.delete_log_group, logGroupName=name)

print()
print('NOTE: SageMaker endpoints still running (billed per invocation only):')
print('  krishirakshak-efficientnet-b3  (ap-south-1)')
print('  bge-m3-krishirakshak           (ap-south-1)')
print('Delete manually in AWS Console if done with the project.')
print()
print('=== Cleanup complete ===')
"
