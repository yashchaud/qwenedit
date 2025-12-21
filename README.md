# Qwen-Image-Layered RunPod

RunPod inference server for [Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered), a revolutionary AI model that decomposes images into multiple RGBA layers for advanced editing capabilities.

**Supports both RunPod Serverless and RunPod Pods deployments!**

## Quick Links

- 📖 **[Complete API Examples](EXAMPLES.md)** - Detailed request/response examples
- ⚖️ **[Deployment Comparison](DEPLOYMENT_COMPARISON.md)** - Serverless vs Pods comparison
- 🚀 **[Deployment Guide](#deployment-options)** - Serverless vs Pods setup
- 🔧 **[Self-Hosted Runner Setup](RUNNER_SETUP.md)** - Build on RunPod CPU instance
- 📚 **[API Documentation](#api-usage)** - Parameter reference

## Features

- Full parameter exposure for Qwen-Image-Layered pipeline
- Variable layer decomposition (configurable layer count)
- Dual output formats: Individual RGBA layers + PowerPoint package
- **Dual deployment modes**: Serverless (on-demand) and Pods (persistent REST API)
- Model caching on RunPod network volume (first start ~2-3 min, subsequent <1s)
- Comprehensive error handling with detailed tracebacks
- GPU-optimized with bfloat16 precision
- Automated Docker builds via GitHub Actions to Docker Hub

## Quick Start

### Docker Image

Images are automatically built and pushed to Docker Hub:

```bash
docker pull <your-dockerhub-username>/qwenedit:latest
```

### Deployment Options

#### Option 1: RunPod Serverless (On-Demand)

Best for sporadic usage, pay-per-second billing.

1. Go to [RunPod Serverless](https://www.runpod.io/serverless-gpu)
2. Create new endpoint
3. Use custom container: `<your-dockerhub-username>/qwenedit:latest`
4. Configure GPU (recommended: RTX 4090 24GB or A100 40GB)
5. Set container disk to 60GB+ (model is ~58GB)
6. Enable network volume for model caching (recommended for faster subsequent starts)
7. Set timeout to 180 seconds (to accommodate first-time model download)
8. Deploy and copy your endpoint URL

#### Option 2: RunPod Pods (Persistent REST API)

Best for consistent usage, always-on REST API with FastAPI interface.

1. Go to [RunPod Pods](https://www.runpod.io/console/gpu-cloud)
2. Deploy a new pod
3. Select GPU (recommended: RTX 4090 24GB or A100 40GB)
4. Use custom container: `<your-dockerhub-username>/qwenedit:latest`
5. Set container disk to 60GB+ (model is ~58GB)
6. **Environment Variables**: Add `DEPLOYMENT_MODE=pod`
7. **Expose HTTP Ports**: Add port `8000` (mapped to public port)
8. Enable network volume for model persistence
9. Deploy and access via: `https://<pod-id>-8000.proxy.runpod.net`

**Pod API Endpoints:**
- `GET /` - API information
- `GET /health` - Health check and model status
- `GET /docs` - Interactive Swagger UI documentation
- `POST /inference` - Main inference endpoint (JSON)
- `POST /inference/upload` - Inference with file upload

## API Usage

**📖 For detailed examples with complete request/response formats, see [EXAMPLES.md](EXAMPLES.md)**

### Serverless API (RunPod Serverless)

#### Input Schema

```json
{
  "input": {
    "image": "base64_encoded_rgba_image or https://url",
    "layers": 4,
    "resolution": 640,
    "true_cfg_scale": 4.0,
    "num_inference_steps": 50,
    "num_images_per_prompt": 1,
    "cfg_normalize": true,
    "use_en_prompt": true,
    "negative_prompt": " ",
    "seed": 777,
    "output_format": "individual"
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | string | **required** | Base64 encoded RGBA PNG or HTTPS URL |
| `layers` | integer | 4 | Number of layers to decompose (1-10 recommended) |
| `resolution` | integer | 640 | Resolution bucket (640 or 1024) |
| `true_cfg_scale` | float | 4.0 | Classifier-free guidance scale (1.0-10.0) |
| `num_inference_steps` | integer | 50 | Number of diffusion steps (10-100) |
| `num_images_per_prompt` | integer | 1 | Number of outputs per prompt (1-4) |
| `cfg_normalize` | boolean | true | Enable CFG normalization |
| `use_en_prompt` | boolean | true | Automatic caption language detection |
| `negative_prompt` | string | " " | Negative prompt guidance |
| `seed` | integer | null | Random seed for reproducibility |
| `output_format` | string | "individual" | Output format: "individual" or "pptx" |

### Output Schema

```json
{
  "layers": [
    {
      "layer_index": 0,
      "image": "base64_rgba_png"
    },
    {
      "layer_index": 1,
      "image": "base64_rgba_png"
    }
  ],
  "package": "base64_pptx (only if output_format=pptx)",
  "metadata": {
    "num_layers": 4,
    "resolution": 640,
    "seed_used": 777
  }
}
```

### Example Usage (Python)

```python
import runpod
import base64
from PIL import Image
from io import BytesIO

# Initialize RunPod
runpod.api_key = "your_api_key"

# Load and encode image
image = Image.open("input.png").convert("RGBA")
buffer = BytesIO()
image.save(buffer, format="PNG")
image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Run inference
endpoint = runpod.Endpoint("your_endpoint_id")
result = endpoint.run_sync({
    "input": {
        "image": image_b64,
        "layers": 4,
        "resolution": 640,
        "seed": 777
    }
})

# Save output layers
for layer_data in result['layers']:
    idx = layer_data['layer_index']
    img_bytes = base64.b64decode(layer_data['image'])
    img = Image.open(BytesIO(img_bytes))
    img.save(f"layer_{idx}.png")
```

#### Example Usage (cURL)

```bash
curl -X POST https://api.runpod.ai/v2/<endpoint_id>/run \
  -H "Authorization: Bearer <your_api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "https://example.com/image.png",
      "layers": 4,
      "resolution": 640,
      "seed": 777
    }
  }'
```

### Pod REST API (RunPod Pods)

When deployed as a Pod with `DEPLOYMENT_MODE=pod`, the server exposes a FastAPI REST interface.

#### Health Check

```bash
curl https://<pod-id>-8000.proxy.runpod.net/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4090"
}
```

#### Inference with JSON

```bash
curl -X POST https://<pod-id>-8000.proxy.runpod.net/inference \
  -H "Content-Type: application/json" \
  -d '{
    "image": "https://example.com/image.png",
    "layers": 4,
    "resolution": 640,
    "seed": 777,
    "output_format": "individual"
  }'
```

#### Inference with File Upload

```bash
curl -X POST https://<pod-id>-8000.proxy.runpod.net/inference/upload \
  -F "file=@input.png" \
  -F "layers=4" \
  -F "resolution=640" \
  -F "seed=777"
```

#### Interactive Documentation

Access Swagger UI at: `https://<pod-id>-8000.proxy.runpod.net/docs`

#### Python Client (Pod)

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Pod endpoint
POD_URL = "https://<pod-id>-8000.proxy.runpod.net"

# Check health
health = requests.get(f"{POD_URL}/health").json()
print(f"Status: {health['status']}, GPU: {health['gpu_name']}")

# Load and encode image
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

## Local Testing

### Prerequisites

- Python 3.10+
- CUDA-capable GPU with 10GB+ VRAM
- Git

### Setup

```bash
# Clone repository
git clone https://github.com/<your-username>/qwenedit.git
cd qwenedit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers.git
pip install runpod

# Add test image
mkdir test_images
# Copy your test image to test_images/sample.png

# Run test
python test_handler.py
```

Output layers will be saved to the `output/` directory.

## Docker Build

### Local Build

```bash
docker build -t qwen-layered .
docker run --gpus all -p 8000:8000 qwen-layered
```

### GitHub Actions

The repository includes automated Docker builds via GitHub Actions:

- **Trigger**: Push to main branch, version tags (v*.*.*), or manual dispatch
- **Output**: Images pushed to GitHub Container Registry (ghcr.io)
- **Tags**: `latest`, semantic versions, commit SHAs

To enable:

1. Ensure GitHub Actions is enabled for your repository
2. Repository must be public or have GHCR permissions configured
3. Push to main branch or create a version tag

## Performance

### Cold Start Time

- **First ever start**: ~2-3 minutes (downloads ~40GB model to RunPod network volume)
- **Subsequent cold starts**: ~30-45 seconds (loads cached model from network volume)
- **Warm container**: <1 second (model cached in GPU memory)

### Inference Time

| Configuration | Estimated Time |
|---------------|----------------|
| 4 layers, 640 resolution, 50 steps | ~15-25 seconds |
| 4 layers, 1024 resolution, 50 steps | ~30-40 seconds |
| 4 layers, 640 resolution, 100 steps | ~30-45 seconds |

### Resource Requirements

| Resolution | VRAM Required | Recommended GPU |
|------------|---------------|-----------------|
| 640 | 8-10 GB | RTX 4090 (24GB) |
| 1024 | 12-16 GB | A100 (40GB) |

## Troubleshooting

### Out of Memory (OOM)

- Reduce `resolution` from 1024 to 640
- Reduce `num_inference_steps`
- Use smaller `num_images_per_prompt` (1 instead of 4)
- Ensure RunPod GPU has sufficient VRAM

### Slow Inference

- Verify GPU is being used (check RunPod logs)
- Increase `num_inference_steps` gradually if quality is poor
- Use 640 resolution for faster inference

### Image Input Errors

- Ensure image is RGBA format (handler auto-converts but source should be compatible)
- For URLs, ensure image is publicly accessible
- For base64, ensure proper encoding without data URL prefix

## Architecture

```
qwenedit/
├── .github/
│   └── workflows/
│       └── docker-build.yml           # CI/CD pipeline (Docker Hub)
├── utils/
│   ├── __init__.py
│   └── pptx_packager.py               # PowerPoint layer packaging
├── handler.py                          # RunPod serverless handler
├── api_server.py                       # FastAPI server for Pods
├── entrypoint.sh                       # Deployment mode selector
├── Dockerfile                          # Multi-mode container definition
├── requirements.txt                    # Python dependencies
├── test_handler.py                     # Local testing script (serverless)
├── test_pod_api.py                     # Local testing script (pods)
├── .env.example                        # Configuration template
├── .gitignore                          # Git ignore rules
├── README.md                           # Main documentation
├── EXAMPLES.md                         # Complete API examples
├── DEPLOYMENT_COMPARISON.md            # Serverless vs Pods comparison
└── RUNNER_SETUP.md                     # Self-hosted runner setup guide
```

### Deployment Modes

The same Docker image supports both deployment modes:

| Mode | Entry Point | Use Case | Billing |
|------|-------------|----------|---------|
| **Serverless** | `handler.py` (RunPod SDK) | Sporadic inference | Pay per second of usage |
| **Pod** | `api_server.py` (FastAPI) | Persistent API, high traffic | Pay per hour |

Controlled by `DEPLOYMENT_MODE` environment variable:
- Default (or `DEPLOYMENT_MODE=serverless`): RunPod Serverless handler
- `DEPLOYMENT_MODE=pod`: FastAPI REST server on port 8000

## Resources

- [Qwen-Image-Layered Model](https://huggingface.co/Qwen/Qwen-Image-Layered)
- [Research Paper](https://arxiv.org/abs/2512.15603)
- [Interactive Demo](https://huggingface.co/spaces/Qwen/Qwen-Image-Layered)
- [RunPod Documentation](https://docs.runpod.io/)
- [Diffusers Library](https://github.com/huggingface/diffusers)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
