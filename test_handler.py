"""
Local Testing Script for RunPod Handler

Tests the handler locally before deploying to RunPod.
"""

import json
import base64
from PIL import Image
from io import BytesIO
import sys
import os

# Import handler function
sys.path.insert(0, '.')
from handler import handler


def test_local():
    """Test handler with local image"""

    # Check if test image exists
    if not os.path.exists("test_images"):
        print("Creating test_images directory...")
        os.makedirs("test_images")
        print("Please add a sample.png image to the test_images directory and run again.")
        return

    test_image_path = "test_images/sample.png"
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}")
        print("Please add a sample.png image to the test_images directory.")
        return

    # Load test image
    print(f"Loading test image from {test_image_path}...")
    test_image = Image.open(test_image_path).convert("RGBA")

    # Encode to base64
    buffer = BytesIO()
    test_image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Create test job
    test_job = {
        "input": {
            "image": image_b64,
            "layers": 4,
            "resolution": 640,
            "true_cfg_scale": 4.0,
            "num_inference_steps": 50,
            "seed": 777,
            "output_format": "individual"
        }
    }

    # Run handler
    print("Running handler...")
    print(f"Parameters: {json.dumps(test_job['input'], indent=2)}")
    result = handler(test_job)

    # Check for errors
    if "error" in result:
        print(f"\nError: {result['error']}")
        print(f"\nTraceback:\n{result['traceback']}")
        return

    # Create output directory
    if not os.path.exists("output"):
        os.makedirs("output")

    # Save output layers
    print(f"\nSuccess! Generated {len(result['layers'])} layers")
    for layer_data in result['layers']:
        idx = layer_data['layer_index']
        img_bytes = base64.b64decode(layer_data['image'])
        img = Image.open(BytesIO(img_bytes))
        output_path = f"output/layer_{idx}.png"
        img.save(output_path)
        print(f"Saved {output_path}")

    print(f"\nMetadata: {json.dumps(result['metadata'], indent=2)}")

    # Save PPTX if present
    if "package" in result:
        pptx_bytes = base64.b64decode(result['package'])
        with open("output/layers.pptx", "wb") as f:
            f.write(pptx_bytes)
        print("Saved output/layers.pptx")


if __name__ == "__main__":
    test_local()
