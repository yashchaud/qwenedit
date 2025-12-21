"""
Test script for Pod API (FastAPI server)

Usage:
    python test_pod_api.py <pod_url>

Example:
    python test_pod_api.py https://abc123-8000.proxy.runpod.net
"""

import sys
import requests
import base64
from PIL import Image
from io import BytesIO
import os

def test_health(pod_url: str):
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{pod_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_inference(pod_url: str, image_path: str):
    """Test inference endpoint with JSON"""
    print(f"Testing inference with {image_path}...")

    # Load and encode image
    image = Image.open(image_path).convert("RGBA")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Make request
    response = requests.post(
        f"{pod_url}/inference",
        json={
            "image": image_b64,
            "layers": 4,
            "resolution": 640,
            "seed": 777,
            "output_format": "individual"
        }
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Generated {len(result['layers'])} layers")
        print(f"Metadata: {result['metadata']}")

        # Save layers
        os.makedirs("output_pod", exist_ok=True)
        for layer_data in result['layers']:
            idx = layer_data['layer_index']
            img_bytes = base64.b64decode(layer_data['image'])
            img = Image.open(BytesIO(img_bytes))
            output_path = f"output_pod/layer_{idx}.png"
            img.save(output_path)
            print(f"Saved {output_path}")
    else:
        print(f"Error: {response.text}")
    print()

def test_upload(pod_url: str, image_path: str):
    """Test inference/upload endpoint with file upload"""
    print(f"Testing file upload with {image_path}...")

    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'layers': 4,
            'resolution': 640,
            'seed': 777,
            'output_format': 'individual'
        }

        response = requests.post(
            f"{pod_url}/inference/upload",
            files=files,
            data=data
        )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Generated {len(result['layers'])} layers")
        print(f"Metadata: {result['metadata']}")

        # Save layers
        os.makedirs("output_pod_upload", exist_ok=True)
        for layer_data in result['layers']:
            idx = layer_data['layer_index']
            img_bytes = base64.b64decode(layer_data['image'])
            img = Image.open(BytesIO(img_bytes))
            output_path = f"output_pod_upload/layer_{idx}.png"
            img.save(output_path)
            print(f"Saved {output_path}")
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pod_api.py <pod_url> [image_path]")
        print("Example: python test_pod_api.py https://abc123-8000.proxy.runpod.net")
        sys.exit(1)

    pod_url = sys.argv[1].rstrip('/')
    image_path = sys.argv[2] if len(sys.argv) > 2 else "test_images/sample.png"

    print(f"Pod URL: {pod_url}")
    print(f"Image: {image_path}")
    print()

    # Test health
    test_health(pod_url)

    # Test inference if image exists
    if os.path.exists(image_path):
        test_inference(pod_url, image_path)
        test_upload(pod_url, image_path)
    else:
        print(f"Image not found: {image_path}")
        print("Skipping inference tests")
