# API Examples - Request & Response Formats

This document provides detailed examples of requests and responses for both RunPod Serverless and Pod deployments.

## Table of Contents

- [Serverless Examples](#serverless-examples)
- [Pod Examples](#pod-examples)
- [Common Parameters](#common-parameters)
- [Output Formats](#output-formats)

---

## Serverless Examples

### Example 1: Basic Inference (Base64 Image)

**Request:**
```json
{
  "input": {
    "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    "layers": 4,
    "resolution": 640
  }
}
```

**Response:**
```json
{
  "delayTime": 15234,
  "executionTime": 14891,
  "id": "sync-abc123-xyz789",
  "output": {
    "layers": [
      {
        "layer_index": 0,
        "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
      },
      {
        "layer_index": 1,
        "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
      },
      {
        "layer_index": 2,
        "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
      },
      {
        "layer_index": 3,
        "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
      }
    ],
    "metadata": {
      "num_layers": 4,
      "resolution": 640,
      "seed_used": null
    }
  },
  "status": "COMPLETED"
}
```

### Example 2: Inference with URL Image

**Request:**
```json
{
  "input": {
    "image": "https://example.com/sample-image.png",
    "layers": 6,
    "resolution": 1024,
    "true_cfg_scale": 5.0,
    "num_inference_steps": 75,
    "seed": 42
  }
}
```

**Response:**
```json
{
  "delayTime": 28456,
  "executionTime": 27923,
  "id": "sync-def456-uvw012",
  "output": {
    "layers": [
      {
        "layer_index": 0,
        "image": "iVBORw0KGgoAAAANSUhEUgAABAAAAAAUACAYAAAAqDc3BAAAA..."
      },
      {
        "layer_index": 1,
        "image": "iVBORw0KGgoAAAANSUhEUgAABAAAAAAUACAYAAAAqDc3BAAAA..."
      },
      {
        "layer_index": 2,
        "image": "iVBORw0KGgoAAAANSUhEUgAABAAAAAAUACAYAAAAqDc3BAAAA..."
      },
      {
        "layer_index": 3,
        "image": "iVBORw0KGgoAAAANSUhEUgAABAAAAAAUACAYAAAAqDc3BAAAA..."
      },
      {
        "layer_index": 4,
        "image": "iVBORw0KGgoAAAANSUhEUgAABAAAAAAUACAYAAAAqDc3BAAAA..."
      },
      {
        "layer_index": 5,
        "image": "iVBORw0KGgoAAAANSUhEUgAABAAAAAAUACAYAAAAqDc3BAAAA..."
      }
    ],
    "metadata": {
      "num_layers": 6,
      "resolution": 1024,
      "seed_used": 42
    }
  },
  "status": "COMPLETED"
}
```

### Example 3: PowerPoint Package Output

**Request:**
```json
{
  "input": {
    "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    "layers": 3,
    "resolution": 640,
    "output_format": "pptx",
    "seed": 777
  }
}
```

**Response:**
```json
{
  "delayTime": 12543,
  "executionTime": 12234,
  "id": "sync-ghi789-rst345",
  "output": {
    "layers": [
      {
        "layer_index": 0,
        "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
      },
      {
        "layer_index": 1,
        "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
      },
      {
        "layer_index": 2,
        "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
      }
    ],
    "package": "UEsDBBQAAAAIAL2MvFYAAAAAAAAAAAAAAAATABwAW0NvbnRlbnRfVHlwZXNdLnhtbCCo...",
    "metadata": {
      "num_layers": 3,
      "resolution": 640,
      "seed_used": 777
    }
  },
  "status": "COMPLETED"
}
```

### Example 4: Advanced Parameters

**Request:**
```json
{
  "input": {
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    "layers": 5,
    "resolution": 640,
    "true_cfg_scale": 3.5,
    "num_inference_steps": 100,
    "num_images_per_prompt": 1,
    "cfg_normalize": true,
    "use_en_prompt": true,
    "negative_prompt": "blurry, low quality, artifacts",
    "seed": 12345,
    "output_format": "individual"
  }
}
```

**Response:**
```json
{
  "delayTime": 18765,
  "executionTime": 18321,
  "id": "sync-jkl012-mno678",
  "output": {
    "layers": [
      {
        "layer_index": 0,
        "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
      },
      {
        "layer_index": 1,
        "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
      },
      {
        "layer_index": 2,
        "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
      },
      {
        "layer_index": 3,
        "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
      },
      {
        "layer_index": 4,
        "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
      }
    ],
    "metadata": {
      "num_layers": 5,
      "resolution": 640,
      "seed_used": 12345
    }
  },
  "status": "COMPLETED"
}
```

### Example 5: Error Response

**Request (Invalid Image):**
```json
{
  "input": {
    "image": "invalid_base64_string",
    "layers": 4
  }
}
```

**Response:**
```json
{
  "delayTime": 123,
  "executionTime": 87,
  "id": "sync-error-abc123",
  "output": {
    "error": "Incorrect padding",
    "traceback": "Traceback (most recent call last):\n  File \"/app/handler.py\", line 78, in handler\n    image = decode_image(job_input[\"image\"])\n  File \"/app/handler.py\", line 152, in decode_image\n    img_bytes = base64.b64decode(image_data)\n  File \"/opt/conda/lib/python3.10/base64.py\", line 87, in b64decode\n    return binascii.a2b_base64(s)\nbinascii.Error: Incorrect padding\n"
  },
  "status": "COMPLETED"
}
```

---

## Pod Examples

### Example 1: Health Check

**Request:**
```bash
GET https://abc123-8000.proxy.runpod.net/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4090"
}
```

### Example 2: Basic Inference (JSON)

**Request:**
```bash
POST https://abc123-8000.proxy.runpod.net/inference
Content-Type: application/json
```

```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "layers": 4,
  "resolution": 640,
  "seed": 777
}
```

**Response:**
```json
{
  "layers": [
    {
      "layer_index": 0,
      "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
    },
    {
      "layer_index": 1,
      "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
    },
    {
      "layer_index": 2,
      "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
    },
    {
      "layer_index": 3,
      "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
    }
  ],
  "metadata": {
    "num_layers": 4,
    "resolution": 640,
    "seed_used": 777
  }
}
```

### Example 3: File Upload

**Request:**
```bash
POST https://abc123-8000.proxy.runpod.net/inference/upload
Content-Type: multipart/form-data
```

**Form Data:**
```
file: [binary PNG file]
layers: 4
resolution: 640
true_cfg_scale: 4.0
num_inference_steps: 50
seed: 42
output_format: individual
```

**Response:**
```json
{
  "layers": [
    {
      "layer_index": 0,
      "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
    },
    {
      "layer_index": 1,
      "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
    },
    {
      "layer_index": 2,
      "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
    },
    {
      "layer_index": 3,
      "image": "iVBORw0KGgoAAAANSUhEUgAABQAAAAPACAYAAAABwHqJAAAA..."
    }
  ],
  "metadata": {
    "num_layers": 4,
    "resolution": 640,
    "seed_used": 42
  }
}
```

### Example 4: PowerPoint Package (Pod)

**Request:**
```bash
POST https://abc123-8000.proxy.runpod.net/inference
Content-Type: application/json
```

```json
{
  "image": "https://example.com/image.png",
  "layers": 3,
  "resolution": 1024,
  "output_format": "pptx",
  "seed": 999
}
```

**Response:**
```json
{
  "layers": [
    {
      "layer_index": 0,
      "image": "iVBORw0KGgoAAAANSUhEUgAABAAAAAAUACAYAAAAqDc3BAAAA..."
    },
    {
      "layer_index": 1,
      "image": "iVBORw0KGgoAAAANSUhEUgAABAAAAAAUACAYAAAAqDc3BAAAA..."
    },
    {
      "layer_index": 2,
      "image": "iVBORw0KGgoAAAANSUhEUgAABAAAAAAUACAYAAAAqDc3BAAAA..."
    }
  ],
  "package": "UEsDBBQAAAAIAL2MvFYAAAAAAAAAAAAAAAATABwAW0NvbnRlbnRfVHlwZXNdLnhtbCCo...",
  "metadata": {
    "num_layers": 3,
    "resolution": 1024,
    "seed_used": 999
  }
}
```

### Example 5: Error Response (Pod)

**Request:**
```bash
POST https://abc123-8000.proxy.runpod.net/inference
Content-Type: application/json
```

```json
{
  "image": "not_a_valid_image",
  "layers": 4
}
```

**Response (HTTP 500):**
```json
{
  "detail": {
    "error": "Incorrect padding",
    "traceback": "Traceback (most recent call last):\n  File \"/app/api_server.py\", line 95, in inference\n    image = decode_image(request.image)\n  File \"/app/api_server.py\", line 210, in decode_image\n    img_bytes = base64.b64decode(image_data)\n  File \"/opt/conda/lib/python3.10/base64.py\", line 87, in b64decode\n    return binascii.a2b_base64(s)\nbinascii.Error: Incorrect padding\n"
  }
}
```

### Example 6: Model Not Loaded (Pod)

**Request:**
```bash
POST https://abc123-8000.proxy.runpod.net/inference
```

**Response (HTTP 503):**
```json
{
  "detail": "Model not loaded yet"
}
```

---

## Common Parameters

### Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `image` | string | Base64 encoded PNG or HTTPS URL | `"iVBORw0..."` or `"https://..."` |

### Optional Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `layers` | integer | 4 | 1-20 | Number of layers to decompose |
| `resolution` | integer | 640 | 640, 1024 | Resolution bucket |
| `true_cfg_scale` | float | 4.0 | 0.0-20.0 | Classifier-free guidance scale |
| `num_inference_steps` | integer | 50 | 1-200 | Number of diffusion steps |
| `num_images_per_prompt` | integer | 1 | 1-4 | Images per prompt (usually 1) |
| `cfg_normalize` | boolean | true | - | Enable CFG normalization |
| `use_en_prompt` | boolean | true | - | Auto caption language detection |
| `negative_prompt` | string | `" "` | - | Negative prompt guidance |
| `seed` | integer | null | - | Random seed for reproducibility |
| `output_format` | string | `"individual"` | `"individual"`, `"pptx"` | Output format |

---

## Output Formats

### Individual Layers (Default)

When `output_format="individual"` (default), returns array of base64-encoded RGBA PNG images.

```json
{
  "layers": [
    {
      "layer_index": 0,
      "image": "base64_rgba_png..."
    }
  ],
  "metadata": {...}
}
```

**To decode in Python:**
```python
import base64
from PIL import Image
from io import BytesIO

for layer_data in result['layers']:
    img_bytes = base64.b64decode(layer_data['image'])
    img = Image.open(BytesIO(img_bytes))
    img.save(f"layer_{layer_data['layer_index']}.png")
```

**To decode in JavaScript:**
```javascript
const layers = result.layers;
layers.forEach(layer => {
  const imgData = `data:image/png;base64,${layer.image}`;
  // Use imgData as src in <img> tag or download
});
```

### PowerPoint Package

When `output_format="pptx"`, returns both individual layers AND a PowerPoint file.

```json
{
  "layers": [...],
  "package": "base64_pptx_file...",
  "metadata": {...}
}
```

**To decode PPTX in Python:**
```python
import base64

pptx_bytes = base64.b64decode(result['package'])
with open('layers.pptx', 'wb') as f:
    f.write(pptx_bytes)
```

**PPTX Structure:**
- Slide 1: Layer 0
- Slide 2: Layer 1
- Slide 3: Layer 2
- etc.

Each layer fills the entire slide (16:9 aspect ratio).

---

## Complete Code Examples

### Python - Serverless

```python
import runpod
import base64
from PIL import Image
from io import BytesIO

# Initialize
runpod.api_key = "your_api_key"
endpoint = runpod.Endpoint("your_endpoint_id")

# Load image
image = Image.open("input.png").convert("RGBA")
buffer = BytesIO()
image.save(buffer, format="PNG")
image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Run inference
result = endpoint.run_sync({
    "input": {
        "image": image_b64,
        "layers": 4,
        "resolution": 640,
        "seed": 777
    }
})

# Save layers
for layer_data in result['output']['layers']:
    idx = layer_data['layer_index']
    img_bytes = base64.b64decode(layer_data['image'])
    img = Image.open(BytesIO(img_bytes))
    img.save(f"layer_{idx}.png")
```

### Python - Pod

```python
import requests
import base64
from PIL import Image
from io import BytesIO

POD_URL = "https://abc123-8000.proxy.runpod.net"

# Load image
image = Image.open("input.png").convert("RGBA")
buffer = BytesIO()
image.save(buffer, format="PNG")
image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Run inference
response = requests.post(
    f"{POD_URL}/inference",
    json={
        "image": image_b64,
        "layers": 4,
        "resolution": 640,
        "seed": 777
    }
)
result = response.json()

# Save layers
for layer_data in result['layers']:
    idx = layer_data['layer_index']
    img_bytes = base64.b64decode(layer_data['image'])
    img = Image.open(BytesIO(img_bytes))
    img.save(f"layer_{idx}.png")
```

### cURL - Serverless

```bash
# Prepare base64 image
IMAGE_B64=$(base64 -w 0 input.png)

# Run inference
curl -X POST https://api.runpod.ai/v2/<endpoint_id>/run \
  -H "Authorization: Bearer <your_api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "'"$IMAGE_B64"'",
      "layers": 4,
      "resolution": 640,
      "seed": 777
    }
  }'
```

### cURL - Pod (JSON)

```bash
# Prepare base64 image
IMAGE_B64=$(base64 -w 0 input.png)

# Run inference
curl -X POST https://abc123-8000.proxy.runpod.net/inference \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'"$IMAGE_B64"'",
    "layers": 4,
    "resolution": 640,
    "seed": 777
  }'
```

### cURL - Pod (File Upload)

```bash
curl -X POST https://abc123-8000.proxy.runpod.net/inference/upload \
  -F "file=@input.png" \
  -F "layers=4" \
  -F "resolution=640" \
  -F "seed=777" \
  -F "output_format=individual"
```

### JavaScript - Pod

```javascript
// Using fetch API
async function runInference(imageFile) {
  // Read file as base64
  const reader = new FileReader();
  reader.readAsDataURL(imageFile);

  reader.onload = async () => {
    const base64Image = reader.result.split(',')[1];

    const response = await fetch('https://abc123-8000.proxy.runpod.net/inference', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        image: base64Image,
        layers: 4,
        resolution: 640,
        seed: 777
      })
    });

    const result = await response.json();

    // Display layers
    result.layers.forEach(layer => {
      const img = document.createElement('img');
      img.src = `data:image/png;base64,${layer.image}`;
      document.body.appendChild(img);
    });
  };
}
```

### JavaScript - Pod (File Upload)

```javascript
async function uploadInference(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('layers', '4');
  formData.append('resolution', '640');
  formData.append('seed', '777');

  const response = await fetch('https://abc123-8000.proxy.runpod.net/inference/upload', {
    method: 'POST',
    body: formData
  });

  const result = await response.json();

  // Display layers
  result.layers.forEach(layer => {
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${layer.image}`;
    document.body.appendChild(img);
  });
}
```

---

## Notes

1. **Image Format**: All output layers are RGBA PNG format with transparency
2. **Base64 Encoding**: Use standard base64 encoding without line breaks
3. **Data URL Prefix**: Optional `data:image/png;base64,` prefix is automatically stripped
4. **HTTPS URLs**: Image URLs must be publicly accessible
5. **Timeouts**: Set appropriate timeouts for serverless (180s recommended)
6. **Error Handling**: Always check for error field in response
7. **Seed Reproducibility**: Same seed + image = same output layers
8. **Resolution**: Use 640 for faster inference, 1024 for higher quality
