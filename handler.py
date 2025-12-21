"""
RunPod Serverless Handler for Qwen-Image-Layered

This handler decomposes images into multiple RGBA layers using Qwen-Image-Layered model.
"""

import runpod
import torch
from diffusers import QwenImageLayeredPipeline
from PIL import Image
import base64
from io import BytesIO
import os
from typing import Dict, Any, List, Optional
import traceback

# Global pipeline instance (loaded once, reused across invocations)
pipeline = None


def load_model():
    """Load model in global scope for container reuse"""
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

    return pipeline


# Preload model at container startup
load_model()


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for Qwen-Image-Layered

    Input Schema (job["input"]):
    {
        "image": str,                    # Base64 encoded RGBA image or URL
        "layers": int,                   # Number of layers (variable)
        "resolution": int,               # 640 or 1024
        "true_cfg_scale": float,         # Default 4.0
        "num_inference_steps": int,      # Default 50
        "num_images_per_prompt": int,    # Default 1
        "cfg_normalize": bool,           # Default True
        "use_en_prompt": bool,           # Default True
        "negative_prompt": str,          # Default " "
        "seed": int,                     # Optional seed for reproducibility
        "output_format": str             # "individual" or "pptx"
    }

    Output Schema:
    {
        "layers": [                      # Individual layer images (base64 RGBA PNGs)
            {"layer_index": 0, "image": "base64..."},
            {"layer_index": 1, "image": "base64..."},
            ...
        ],
        "package": str,                  # Optional: base64 PPTX
        "metadata": {
            "num_layers": int,
            "resolution": int,
            "seed_used": int
        }
    }
    """
    try:
        job_input = job["input"]

        # Decode input image
        image = decode_image(job_input["image"])

        # Ensure RGBA
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Extract parameters with defaults
        params = extract_parameters(job_input)

        # Set up generator for reproducibility
        generator = None
        if params.get("seed") is not None:
            generator = torch.Generator(device='cuda').manual_seed(params["seed"])
            seed_used = params["seed"]
        else:
            seed_used = None

        # Prepare pipeline inputs
        pipeline_inputs = {
            "image": image,
            "generator": generator,
            "true_cfg_scale": params["true_cfg_scale"],
            "negative_prompt": params["negative_prompt"],
            "num_inference_steps": params["num_inference_steps"],
            "num_images_per_prompt": params["num_images_per_prompt"],
            "layers": params["layers"],
            "resolution": params["resolution"],
            "cfg_normalize": params["cfg_normalize"],
            "use_en_prompt": params["use_en_prompt"],
        }

        # Clear cache before inference
        torch.cuda.empty_cache()

        # Run inference
        print(f"Running inference with {params['layers']} layers at {params['resolution']} resolution...")
        with torch.inference_mode():
            output = pipeline(**pipeline_inputs)
            output_layers = output.images[0]  # List of PIL Images (RGBA)

        # Clear cache after inference
        torch.cuda.empty_cache()

        print(f"Inference complete! Generated {len(output_layers)} layers")

        # Format output based on output_format
        result = format_output(
            output_layers,
            params["output_format"],
            params
        )

        # Add metadata
        result["metadata"] = {
            "num_layers": len(output_layers),
            "resolution": params["resolution"],
            "seed_used": seed_used
        }

        return result

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


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


def extract_parameters(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and validate parameters with defaults"""
    return {
        "layers": job_input.get("layers", 4),
        "resolution": job_input.get("resolution", 640),
        "true_cfg_scale": job_input.get("true_cfg_scale", 4.0),
        "num_inference_steps": job_input.get("num_inference_steps", 50),
        "num_images_per_prompt": job_input.get("num_images_per_prompt", 1),
        "cfg_normalize": job_input.get("cfg_normalize", True),
        "use_en_prompt": job_input.get("use_en_prompt", True),
        "negative_prompt": job_input.get("negative_prompt", " "),
        "seed": job_input.get("seed"),
        "output_format": job_input.get("output_format", "individual")
    }


def encode_image(image: Image.Image) -> str:
    """Encode PIL Image to base64 PNG"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def format_output(
    layers: List[Image.Image],
    output_format: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Format output based on requested format"""
    # Always include individual layers
    layer_data = []
    for i, layer in enumerate(layers):
        layer_data.append({
            "layer_index": i,
            "image": encode_image(layer)
        })

    result = {"layers": layer_data}

    # Add packaged format if requested
    if output_format == "pptx":
        from utils.pptx_packager import create_pptx
        pptx_bytes = create_pptx(layers, params)
        result["package"] = base64.b64encode(pptx_bytes).decode('utf-8')

    return result


# Start RunPod serverless
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
