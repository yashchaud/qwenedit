"""
Simple Flask API server for Qwen-Image-Edit pipeline.

Accepts POST requests with:
- ref_image: Reference image (base64 or file)
- masked_image: Masked image showing edit area (base64 or file)
- prompt: Text instruction for editing

Returns edited image as base64 or file.

Usage:
    python server.py

Example request:
    curl -X POST http://localhost:5000/edit \
      -F "ref_image=@input.jpg" \
      -F "masked_image=@mask.jpg" \
      -F "prompt=Remove the object in the masked area"
"""

import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from PIL import Image
import qwen_client

app = Flask(__name__)

# Initialize Qwen client
# Use official Qwen model (compatible with diffusers)
MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen-Image-Edit")
client = qwen_client.QwenClient(
    model_path=MODEL_PATH,
    mode="pipeline",
    device="cuda",
    torch_dtype="float16"
)

# Load pipeline on startup
print("Loading Qwen model...")
client.load_pipeline()
print("Model ready!")


def decode_image(image_data):
    """Decode image from base64 string or file bytes."""
    if isinstance(image_data, str):
        # Base64 string
        if image_data.startswith('data:image'):
            # Remove data URL prefix
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        return Image.open(BytesIO(img_bytes))
    else:
        # File bytes
        return Image.open(BytesIO(image_data))


def encode_image(image):
    """Encode PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str


def combine_ref_and_mask(ref_image, masked_image):
    """
    Combine reference and mask images for Qwen-Image-Edit.
    The model uses both images to understand what to edit.
    """
    # Ensure same size
    if ref_image.size != masked_image.size:
        masked_image = masked_image.resize(ref_image.size, Image.LANCZOS)

    return [ref_image, masked_image]


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": MODEL_PATH})


@app.route('/edit', methods=['POST'])
def edit_image():
    """
    Main image editing endpoint.

    Accepts:
    - ref_image: Reference/input image (file or base64)
    - masked_image: Mask image showing edit area (file or base64)
    - prompt: Text instruction for editing
    - num_inference_steps: Optional, default 50
    - guidance_scale: Optional, default 7.5
    - return_type: 'json' (base64) or 'file' (image file), default 'json'

    Returns:
    - JSON with base64 image, or
    - Image file (PNG)
    """
    try:
        # Get inputs
        prompt = request.form.get('prompt') or request.json.get('prompt')
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        # Get images
        ref_image = None
        masked_image = None

        # Try form files first
        if 'ref_image' in request.files:
            ref_image = decode_image(request.files['ref_image'].read())
        elif 'ref_image' in request.form:
            ref_image = decode_image(request.form['ref_image'])
        elif request.json and 'ref_image' in request.json:
            ref_image = decode_image(request.json['ref_image'])

        if 'masked_image' in request.files:
            masked_image = decode_image(request.files['masked_image'].read())
        elif 'masked_image' in request.form:
            masked_image = decode_image(request.form['masked_image'])
        elif request.json and 'masked_image' in request.json:
            masked_image = decode_image(request.json['masked_image'])

        if not ref_image:
            return jsonify({"error": "ref_image is required"}), 400
        if not masked_image:
            return jsonify({"error": "masked_image is required"}), 400

        # Get optional parameters
        num_inference_steps = int(request.form.get('num_inference_steps', 50))
        guidance_scale = float(request.form.get('guidance_scale', 7.5))
        return_type = request.form.get('return_type', 'json')

        print(f"Processing edit: {prompt}")
        print(f"Ref image size: {ref_image.size}, Mask size: {masked_image.size}")

        # Combine images for pipeline
        images = combine_ref_and_mask(ref_image, masked_image)

        # Generate edited image
        result = client.generate(
            prompt=prompt,
            images=images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            return_image=True
        )

        print("Edit complete!")

        # Return result
        if return_type == 'file':
            buffer = BytesIO()
            result.save(buffer, format="PNG")
            buffer.seek(0)
            return send_file(buffer, mimetype='image/png')
        else:
            # Return as JSON with base64
            return jsonify({
                "success": True,
                "image": encode_image(result),
                "prompt": prompt
            })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/edit_simple', methods=['POST'])
def edit_simple():
    """
    Simplified endpoint - just ref image and prompt.
    Useful when you don't have a separate mask.

    Accepts:
    - image: Input image (file or base64)
    - prompt: Text instruction for editing

    Returns:
    - JSON with base64 edited image
    """
    try:
        prompt = request.form.get('prompt') or request.json.get('prompt')
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        # Get image
        image = None
        if 'image' in request.files:
            image = decode_image(request.files['image'].read())
        elif 'image' in request.form:
            image = decode_image(request.form['image'])
        elif request.json and 'image' in request.json:
            image = decode_image(request.json['image'])

        if not image:
            return jsonify({"error": "image is required"}), 400

        print(f"Processing simple edit: {prompt}")

        # Generate edited image (single image input)
        result = client.generate(
            prompt=prompt,
            images=[image],
            num_inference_steps=50,
            guidance_scale=7.5,
            return_image=True
        )

        return jsonify({
            "success": True,
            "image": encode_image(result),
            "prompt": prompt
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/batch', methods=['POST'])
def batch_edit():
    """
    Batch processing endpoint - multiple images with same prompt.

    Accepts:
    - images: List of images (files or base64 array)
    - prompt: Text instruction for editing

    Returns:
    - JSON with array of base64 edited images
    """
    try:
        prompt = request.form.get('prompt') or request.json.get('prompt')
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        # Get images
        images = []
        if request.files:
            for key in request.files:
                if key.startswith('image'):
                    images.append(decode_image(request.files[key].read()))
        elif request.json and 'images' in request.json:
            for img_data in request.json['images']:
                images.append(decode_image(img_data))

        if not images:
            return jsonify({"error": "images are required"}), 400

        print(f"Batch processing {len(images)} images: {prompt}")

        results = []
        for i, img in enumerate(images):
            print(f"Processing image {i+1}/{len(images)}")
            result = client.generate(
                prompt=prompt,
                images=[img],
                num_inference_steps=50,
                guidance_scale=7.5,
                return_image=True
            )
            results.append(encode_image(result))

        return jsonify({
            "success": True,
            "images": results,
            "count": len(results),
            "prompt": prompt
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run server
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
