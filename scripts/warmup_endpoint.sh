#!/bin/bash
# Warm up SageMaker endpoint before demo (avoids cold start)
set -euo pipefail

ENDPOINT="krishirakshak-florence2"

echo "Warming up SageMaker endpoint: $ENDPOINT"
echo "Sending 3 dummy requests..."

# Create a small test image (1x1 white pixel JPEG)
python3 -c "
import base64, io, json, boto3
from PIL import Image

img = Image.new('RGB', (224, 224), (0, 128, 0))
buf = io.BytesIO()
img.save(buf, 'JPEG')
img_b64 = base64.b64encode(buf.getvalue()).decode()

client = boto3.client('sagemaker-runtime')
for i in range(3):
    try:
        resp = client.invoke_endpoint(
            EndpointName='$ENDPOINT',
            ContentType='application/json',
            Body=json.dumps({'image': img_b64, 'prompt': '<CROP_DISEASE>'})
        )
        print(f'  Request {i+1}: OK')
    except Exception as e:
        print(f'  Request {i+1}: {e}')
"

echo "Endpoint warm! Ready for demo."
