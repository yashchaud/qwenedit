"""
Test script for Job-based API
"""

import sys
import requests
import base64
from PIL import Image
from io import BytesIO
import time
import os

def create_job(pod_url: str, image_path: str):
    """Create an inference job"""
    print(f"Creating inference job with {image_path}...")

    # Load and encode image
    image = Image.open(image_path).convert("RGBA")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Create job
    response = requests.post(
        f"{pod_url}/inference/job",
        json={
            "image": image_b64,
            "layers": 4,
            "resolution": 640,
            "seed": 777
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Job created: {result['job_id']}")
        print(f"Status: {result['status']}")
        return result['job_id']
    else:
        print(f"Error creating job: {response.text}")
        return None

def check_job_status(pod_url: str, job_id: str):
    """Check job status"""
    response = requests.get(f"{pod_url}/job/{job_id}")

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error checking status: {response.text}")
        return None

def download_layers(pod_url: str, job_id: str, output_dir: str = "output_job"):
    """Download all layers for a completed job"""
    # Check job status
    status_data = check_job_status(pod_url, job_id)

    if not status_data:
        return False

    if status_data['status'] != 'completed':
        print(f"Job is not completed yet. Status: {status_data['status']}")
        return False

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Download each layer
    num_layers = len(status_data['layers'])
    print(f"Downloading {num_layers} layers...")

    for i in range(num_layers):
        response = requests.get(f"{pod_url}/download/{job_id}/{i}")

        if response.status_code == 200:
            output_path = os.path.join(output_dir, f"layer_{i}.png")
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {output_path}")
        else:
            print(f"Error downloading layer {i}: {response.text}")

    print(f"\nMetadata: {status_data['metadata']}")
    return True

def wait_for_job(pod_url: str, job_id: str, timeout: int = 300):
    """Wait for job to complete with polling"""
    print(f"Waiting for job {job_id} to complete...")

    start_time = time.time()

    while True:
        status_data = check_job_status(pod_url, job_id)

        if not status_data:
            return False

        status = status_data['status']
        print(f"Status: {status}")

        if status == 'completed':
            print("Job completed successfully!")
            return True
        elif status == 'failed':
            print(f"Job failed: {status_data.get('error')}")
            return False

        # Check timeout
        if time.time() - start_time > timeout:
            print("Timeout waiting for job")
            return False

        # Wait before next poll
        time.sleep(5)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_job_api.py <pod_url> [image_path]")
        print("Example: python test_job_api.py https://abc123-8000.proxy.runpod.net")
        sys.exit(1)

    pod_url = sys.argv[1].rstrip('/')
    image_path = sys.argv[2] if len(sys.argv) > 2 else "test_images/sample.webp"

    print(f"Pod URL: {pod_url}")
    print(f"Image: {image_path}")
    print()

    # Create job
    job_id = create_job(pod_url, image_path)

    if not job_id:
        sys.exit(1)

    print()

    # Wait for completion
    if wait_for_job(pod_url, job_id):
        print()
        # Download results
        download_layers(pod_url, job_id)
