"""
FastAPI Server for RunPod Pods (Persistent Deployment)

This server exposes the Qwen-Image-Layered model as a REST API for RunPod Pods.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import torch
from diffusers import QwenImageLayeredPipeline
from PIL import Image
import base64
from io import BytesIO
import os
import traceback
import tempfile
import uvicorn
import uuid
import time
from pathlib import Path

# Initialize FastAPI
app = FastAPI(
    title="Qwen-Image-Layered API",
    description="Image layer decomposition using Qwen-Image-Layered model",
    version="1.0.0"
)

# Global pipeline instance
pipeline = None

# Job storage (in production, use Redis or database)
jobs = {}

# Output directory for saved files
OUTPUT_DIR = Path("/root/qwenedit/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class InferenceRequest(BaseModel):
    """Request schema for inference endpoint"""
    image: str = Field(..., description="Base64 encoded RGBA image or URL")
    layers: int = Field(4, description="Number of layers (variable)", ge=1, le=20)
    resolution: int = Field(640, description="Resolution: 640 or 1024")
    true_cfg_scale: float = Field(4.0, description="Guidance scale", ge=0.0, le=20.0)
    num_inference_steps: int = Field(50, description="Number of inference steps", ge=1, le=200)
    num_images_per_prompt: int = Field(1, description="Number of images per prompt", ge=1, le=4)
    cfg_normalize: bool = Field(True, description="Normalize CFG")
    use_en_prompt: bool = Field(True, description="Use English prompt")
    negative_prompt: str = Field(" ", description="Negative prompt")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    output_format: str = Field("individual", description="Output format: 'individual' or 'pptx'")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None


@app.on_event("startup")
async def load_model():
    """Load model at startup"""
    global pipeline
    if pipeline is None:
        print("Loading Qwen-Image-Layered pipeline...")

        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Load with memory optimizations
        pipeline = QwenImageLayeredPipeline.from_pretrained(
            "Qwen/Qwen-Image-Layered",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        pipeline = pipeline.to("cuda")

        # Enable memory efficient attention if available
        try:
            pipeline.enable_attention_slicing(1)
            print("Attention slicing enabled")
        except:
            pass

        # Enable VAE slicing to reduce memory
        try:
            pipeline.enable_vae_slicing()
            print("VAE slicing enabled")
        except:
            pass

        pipeline.set_progress_bar_config(disable=True)

        # Clear cache after loading
        torch.cuda.empty_cache()

        # Print memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"Pipeline loaded successfully!")
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Qwen-Image-Layered API Server",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return HealthResponse(
        status="healthy" if pipeline is not None else "loading",
        model_loaded=pipeline is not None,
        gpu_available=gpu_available,
        gpu_name=gpu_name
    )


@app.post("/inference/job")
async def create_inference_job(request: InferenceRequest, background_tasks: BackgroundTasks):
    """
    Create an inference job that runs in the background

    Returns a job_id immediately. Use /job/{job_id} to check status and get results.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Initialize job status
    jobs[job_id] = {
        "status": "queued",
        "created_at": time.time(),
        "completed_at": None,
        "error": None,
        "layers": [],
        "metadata": {}
    }

    # Start background task
    background_tasks.add_task(run_inference_job, job_id, request)

    return JSONResponse(content={
        "job_id": job_id,
        "status": "queued",
        "message": "Job created successfully. Use GET /job/{job_id} to check status."
    })


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status and results of an inference job
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Build response
    response = {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"]
    }

    if job["status"] == "completed":
        response["completed_at"] = job["completed_at"]
        response["layers"] = job["layers"]
        response["metadata"] = job["metadata"]
    elif job["status"] == "failed":
        response["error"] = job["error"]

    return JSONResponse(content=response)


@app.get("/download/{job_id}/{layer_index}")
async def download_layer(job_id: str, layer_index: int):
    """
    Download a specific layer file by job_id and layer_index
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {job['status']}, not completed")

    if layer_index >= len(job["layers"]):
        raise HTTPException(status_code=404, detail=f"Layer {layer_index} not found")

    file_path = job["layers"][layer_index]["file_path"]

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        file_path,
        media_type="image/png",
        filename=f"layer_{layer_index}.png"
    )


@app.post("/inference")
async def inference(request: InferenceRequest):
    """
    Main inference endpoint for layer decomposition

    Accepts a base64 image or URL and returns decomposed layers
    """
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        # Decode input image
        image = decode_image(request.image)

        # Ensure RGBA
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Set up generator for reproducibility
        generator = None
        seed_used = None
        if request.seed is not None:
            generator = torch.Generator(device='cuda').manual_seed(request.seed)
            seed_used = request.seed

        # Prepare pipeline inputs
        pipeline_inputs = {
            "image": image,
            "generator": generator,
            "true_cfg_scale": request.true_cfg_scale,
            "negative_prompt": request.negative_prompt,
            "num_inference_steps": request.num_inference_steps,
            "num_images_per_prompt": request.num_images_per_prompt,
            "layers": request.layers,
            "resolution": request.resolution,
            "cfg_normalize": request.cfg_normalize,
            "use_en_prompt": request.use_en_prompt,
        }

        # Clear cache before inference
        torch.cuda.empty_cache()

        # Run inference
        print(f"Running inference with {request.layers} layers at {request.resolution} resolution...")
        with torch.inference_mode():
            output = pipeline(**pipeline_inputs)
            output_layers = output.images[0]  # List of PIL Images (RGBA)

        # Clear cache after inference
        torch.cuda.empty_cache()

        print(f"Inference complete! Generated {len(output_layers)} layers")

        # Format output
        layer_data = []
        for i, layer in enumerate(output_layers):
            layer_data.append({
                "layer_index": i,
                "image": encode_image(layer)
            })

        result = {"layers": layer_data}

        # Add packaged format if requested
        if request.output_format == "pptx":
            from utils.pptx_packager import create_pptx
            pptx_bytes = create_pptx(output_layers, request.dict())
            result["package"] = base64.b64encode(pptx_bytes).decode('utf-8')

        # Add metadata
        result["metadata"] = {
            "num_layers": len(output_layers),
            "resolution": request.resolution,
            "seed_used": seed_used
        }

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@app.post("/inference/upload")
async def inference_upload(
    file: UploadFile = File(...),
    layers: int = 4,
    resolution: int = 640,
    true_cfg_scale: float = 4.0,
    num_inference_steps: int = 50,
    num_images_per_prompt: int = 1,
    cfg_normalize: bool = True,
    use_en_prompt: bool = True,
    negative_prompt: str = " ",
    seed: Optional[int] = None,
    output_format: str = "individual"
):
    """
    Inference endpoint with file upload

    Accepts an uploaded image file instead of base64
    """
    try:
        # Read uploaded file
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        # Convert to base64 for processing
        image_b64 = encode_image(image)

        # Create request
        request = InferenceRequest(
            image=image_b64,
            layers=layers,
            resolution=resolution,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            cfg_normalize=cfg_normalize,
            use_en_prompt=use_en_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            output_format=output_format
        )

        # Process through main inference endpoint
        return await inference(request)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


def decode_image(image_data: str) -> Image.Image:
    """Decode image from base64 or URL"""
    if image_data.startswith('http://') or image_data.startswith('https://'):
        import requests
        response = requests.get(image_data)
        return Image.open(BytesIO(response.content))
    else:
        # Base64
        if image_data.startswith('data:image'):
            # Remove data URL prefix if present
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        return Image.open(BytesIO(img_bytes))


def encode_image(image: Image.Image) -> str:
    """Encode PIL Image to base64 PNG"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def run_inference_job(job_id: str, request: InferenceRequest):
    """
    Background task to run inference and save results to disk
    """
    try:
        jobs[job_id]["status"] = "processing"

        # Decode input image
        image = decode_image(request.image)

        # Ensure RGBA
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Set up generator for reproducibility
        generator = None
        seed_used = None
        if request.seed is not None:
            generator = torch.Generator(device='cuda').manual_seed(request.seed)
            seed_used = request.seed

        # Prepare pipeline inputs
        pipeline_inputs = {
            "image": image,
            "generator": generator,
            "true_cfg_scale": request.true_cfg_scale,
            "negative_prompt": request.negative_prompt,
            "num_inference_steps": request.num_inference_steps,
            "num_images_per_prompt": request.num_images_per_prompt,
            "layers": request.layers,
            "resolution": request.resolution,
            "cfg_normalize": request.cfg_normalize,
            "use_en_prompt": request.use_en_prompt,
        }

        # Clear cache before inference
        torch.cuda.empty_cache()

        # Run inference
        print(f"[Job {job_id}] Running inference with {request.layers} layers at {request.resolution} resolution...")
        with torch.inference_mode():
            output = pipeline(**pipeline_inputs)
            output_layers = output.images[0]  # List of PIL Images (RGBA)

        # Clear cache after inference
        torch.cuda.empty_cache()

        print(f"[Job {job_id}] Inference complete! Generated {len(output_layers)} layers")

        # Create job directory
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Save layers to disk
        layer_info = []
        for i, layer in enumerate(output_layers):
            file_path = job_dir / f"layer_{i}.png"
            layer.save(file_path, format="PNG")
            layer_info.append({
                "layer_index": i,
                "file_path": str(file_path),
                "filename": f"layer_{i}.png"
            })

        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = time.time()
        jobs[job_id]["layers"] = layer_info
        jobs[job_id]["metadata"] = {
            "num_layers": len(output_layers),
            "resolution": request.resolution,
            "seed_used": seed_used
        }

        print(f"[Job {job_id}] Job completed successfully. Files saved to {job_dir}")

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = {
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"[Job {job_id}] Job failed: {str(e)}")
        print(traceback.format_exc())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
