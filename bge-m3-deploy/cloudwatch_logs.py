from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

import boto3


REGION = "ap-south-1"
ENDPOINT_NAME = "krishirakshak-bge-m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect recent CloudWatch logs for a SageMaker endpoint.")
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--endpoint-name", default=ENDPOINT_NAME)
    parser.add_argument("--since-minutes", type=int, default=30)
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_group = f"/aws/sagemaker/Endpoints/{args.endpoint_name}"
    start_time = int((datetime.now(UTC) - timedelta(minutes=args.since_minutes)).timestamp() * 1000)

    logs = boto3.client("logs", region_name=args.region)
    streams_response = logs.describe_log_streams(
        logGroupName=log_group,
        orderBy="LastEventTime",
        descending=True,
        limit=5,
    )
    streams = streams_response.get("logStreams", [])
    if not streams:
        print(f"No log streams found in {log_group}")
        return

    for stream in streams:
        stream_name = stream["logStreamName"]
        print(f"=== {stream_name} ===")
        events_response = logs.get_log_events(
            logGroupName=log_group,
            logStreamName=stream_name,
            startTime=start_time,
            limit=args.limit,
            startFromHead=False,
        )
        events = events_response.get("events", [])
        if not events:
            print("No recent events")
            continue
        for event in events:
            timestamp = datetime.fromtimestamp(event["timestamp"] / 1000, UTC).isoformat()
            message = event["message"].encode("ascii", "backslashreplace").decode("ascii")
            print(f"{timestamp} {message}")


if __name__ == "__main__":
    main()
