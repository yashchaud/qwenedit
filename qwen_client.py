"""
Qwen Client - Python client for Qwen-Image-Edit models

Supports:
- Local pipeline execution (diffusers/transformers)
- Remote API endpoints (RunPod, etc.)
- Local vLLM server management
- Multiple image inputs (1-3 images)

Usage:
    import qwen_client

    # Pipeline mode
    client = qwen_client.QwenClient(
        model_path="Qwen/Qwen-Image-Edit-2509",
        mode="pipeline"
    )
    result = client.generate("Make sky vibrant", images=["input.jpg"])
    client.save_image(result, "output.jpg")

    # Remote endpoint
    client = qwen_client.QwenClient(endpoint="https://your-pod.runpod.net")
    result = client.generate("Add sunglasses", images=["portrait.jpg"])

    # Server mode
    with qwen_client.QwenClient(model_path="Qwen/Qwen-Image-Edit-2509") as client:
        client.start_server()
        result = client.generate("Enhance quality", images=["photo.jpg"])
"""

import os
import subprocess
import time
import atexit
import base64
from io import BytesIO
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import requests


class QwenClient:
    """Client for Qwen models with support for image editing and text generation."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        endpoint: Optional[str] = None,
        mode: str = "server",
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        api_key: Optional[str] = None
    ):
        """
        Initialize Qwen client.

        Args:
            model_path: Model path or HuggingFace ID (defaults to QWEN_MODEL_PATH env)
            endpoint: Remote endpoint URL (defaults to QWEN_ENDPOINT env)
            mode: 'server', 'pipeline', or 'remote'
            device: Device for pipeline mode ('cuda', 'cpu', 'auto')
            torch_dtype: Torch dtype ('float16', 'bfloat16', 'auto')
            trust_remote_code: Whether to trust remote code
            api_key: API key (defaults to QWEN_API_KEY env)
        """
        self.model_path = model_path or os.getenv("QWEN_MODEL_PATH")
        self.endpoint = endpoint or os.getenv("QWEN_ENDPOINT")
        self.api_key = api_key or os.getenv("QWEN_API_KEY")

        # Auto-detect mode
        if mode == "server" and self.endpoint and not self.model_path:
            mode = "remote"

        self.mode = mode
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code

        # State
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port: Optional[int] = None
        self.server_host: str = "127.0.0.1"
        self.pipeline = None
        self.tokenizer = None
        self.model = None

        self._validate_config()
        atexit.register(self.cleanup)

    def _validate_config(self):
        """Validate configuration."""
        if self.mode == "remote" and not self.endpoint:
            raise ValueError("endpoint required for remote mode")
        if self.mode in ("server", "pipeline") and not self.model_path:
            raise ValueError(f"model_path required for {self.mode} mode")

    def start_server(
        self,
        port: int = 8000,
        host: str = "127.0.0.1",
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        enable_lora: bool = False,
        additional_args: Optional[List[str]] = None
    ):
        """Start local vLLM server."""
        if self.mode != "server":
            raise RuntimeError(f"Cannot start server in {self.mode} mode")
        if self.server_process:
            raise RuntimeError("Server already running")

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--host", host,
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--dtype", dtype
        ]

        if max_model_len:
            cmd.extend(["--max-model-len", str(max_model_len)])
        if quantization:
            cmd.extend(["--quantization", quantization])
        if enable_lora:
            cmd.append("--enable-lora")
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if additional_args:
            cmd.extend(additional_args)

        try:
            self.server_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1
            )
            self.server_port = port
            self.server_host = host
            self.endpoint = f"http://{host}:{port}/v1/completions"

            self._wait_for_server()
        except FileNotFoundError:
            raise ImportError("vLLM not found. Install: pip install vllm")
        except Exception as e:
            self.stop_server()
            raise RuntimeError(f"Failed to start server: {e}")

    def _wait_for_server(self, timeout: int = 120):
        """Wait for server to be ready."""
        start = time.time()
        url = f"http://{self.server_host}:{self.server_port}/health"

        while time.time() - start < timeout:
            try:
                if requests.get(url, timeout=1).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)

        raise RuntimeError(f"Server failed to start within {timeout}s")

    def stop_server(self):
        """Stop local server."""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            finally:
                self.server_process = None
                self.server_port = None

    def load_pipeline(self):
        """Load model pipeline."""
        if self.mode != "pipeline":
            raise RuntimeError(f"Cannot load pipeline in {self.mode} mode")
        if self.pipeline:
            return

        is_image_model = "image" in self.model_path.lower()
        is_gguf = self.model_path.lower().endswith('.gguf') or 'gguf' in self.model_path.lower()
        is_quantized = any(q in self.model_path.lower() for q in ['4bit', '8bit', 'nf4', 'int4', 'int8'])

        if is_image_model:
            try:
                from diffusers import DiffusionPipeline
                import torch
            except ImportError:
                raise ImportError(
                    "diffusers required. Install: pip install diffusers torch transformers accelerate"
                )

            # Handle GGUF models
            if is_gguf:
                try:
                    from diffusers import GGUFQuantizationConfig
                except ImportError:
                    raise ImportError(
                        "GGUF support requires diffusers>=0.27.0. Install: pip install diffusers>=0.27.0"
                    )

                # Load GGUF quantized model
                self.pipeline = DiffusionPipeline.from_pretrained(
                    self.model_path,
                    quantization_config=GGUFQuantizationConfig(),
                    trust_remote_code=self.trust_remote_code
                )
            # Handle 4-bit/8-bit quantization
            elif is_quantized:
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError:
                    raise ImportError(
                        "Quantization requires bitsandbytes. Install: pip install bitsandbytes"
                    )

                # Detect quantization level
                if '4bit' in self.model_path.lower() or 'nf4' in self.model_path.lower():
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                else:  # 8bit
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)

                self.pipeline = DiffusionPipeline.from_pretrained(
                    self.model_path,
                    quantization_config=quant_config,
                    trust_remote_code=self.trust_remote_code
                )
            # Standard loading (FP16/BF16/FP32)
            else:
                dtype = {"auto": torch.float16, "float16": torch.float16,
                        "bfloat16": torch.bfloat16, "float32": torch.float32}[self.torch_dtype]

                self.pipeline = DiffusionPipeline.from_pretrained(
                    self.model_path, torch_dtype=dtype, trust_remote_code=self.trust_remote_code
                )

            if self.device != "auto" and not is_quantized:
                self.pipeline.to(self.device)
        else:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                import torch
            except ImportError:
                raise ImportError("transformers required. Install: pip install transformers torch")

            dtype = {"auto": "auto", "float16": torch.float16,
                    "bfloat16": torch.bfloat16, "float32": torch.float32}[self.torch_dtype]

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=dtype, device_map=self.device,
                trust_remote_code=self.trust_remote_code
            )
            self.pipeline = pipeline(
                "text-generation", model=self.model, tokenizer=self.tokenizer,
                device_map=self.device
            )

    def _load_image(self, image_path: str):
        """Load image from file or URL."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow required. Install: pip install Pillow")

        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        return Image.open(image_path)

    def _encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64."""
        image = self._load_image(image_path)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def generate(
        self,
        prompt: str,
        images: Optional[Union[str, List[str]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        return_image: bool = True,
        **kwargs
    ) -> Union[str, Dict[str, Any], Any]:
        """
        Generate text or images.

        Args:
            prompt: Text instruction
            images: Image path(s) or URL(s) (supports 1-3 images)
            max_tokens: Max tokens for text generation
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            stop: Stop sequences
            stream: Stream response (server/remote only)
            num_inference_steps: Diffusion steps (image models)
            guidance_scale: Guidance scale (image models)
            return_image: Return PIL Image or bytes
            **kwargs: Additional parameters

        Returns:
            Text string, PIL Image, or image bytes
        """
        if self.mode == "pipeline":
            return self._generate_pipeline(
                prompt, images, max_tokens, temperature, top_p, top_k,
                repetition_penalty, stop, num_inference_steps,
                guidance_scale, return_image, **kwargs
            )
        return self._generate_api(
            prompt, images, max_tokens, temperature, top_p, top_k,
            repetition_penalty, stop, stream, num_inference_steps,
            guidance_scale, return_image, **kwargs
        )

    def _generate_pipeline(
        self, prompt, images, max_tokens, temperature, top_p, top_k,
        repetition_penalty, stop, num_inference_steps, guidance_scale,
        return_image, **kwargs
    ):
        """Generate using pipeline."""
        if not self.pipeline:
            self.load_pipeline()

        if images:
            # Image generation
            if isinstance(images, str):
                images = [images]

            loaded = [self._load_image(img) for img in images]
            config = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "image": loaded[0] if len(loaded) == 1 else loaded,
                **kwargs
            }

            output = self.pipeline(**config)
            result = output.images[0] if hasattr(output, 'images') else output

            if return_image:
                return result
            buffer = BytesIO()
            result.save(buffer, format="PNG")
            return buffer.getvalue()
        else:
            # Text generation
            config = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": temperature > 0,
                "return_full_text": False,
                **kwargs
            }
            if stop:
                config["stop_sequence"] = stop

            outputs = self.pipeline(prompt, **config)
            return outputs[0]["generated_text"] if isinstance(outputs, list) else outputs["generated_text"]

    def _generate_api(
        self, prompt, images, max_tokens, temperature, top_p, top_k,
        repetition_penalty, stop, stream, num_inference_steps,
        guidance_scale, return_image, **kwargs
    ):
        """Generate using API."""
        if not self.endpoint:
            raise RuntimeError("No endpoint configured")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "stream": stream,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            **kwargs
        }

        if images:
            if isinstance(images, str):
                images = [images]

            encoded = []
            for img in images:
                if img.startswith(('http://', 'https://')):
                    encoded.append({"type": "url", "url": img})
                else:
                    encoded.append({"type": "base64", "data": self._encode_image_base64(img)})
            payload["images"] = encoded

        if stop:
            payload["stop"] = stop

        response = requests.post(self.endpoint, json=payload, headers=headers,
                               timeout=120, stream=stream)
        response.raise_for_status()

        if stream:
            return response.iter_lines()

        data = response.json()

        # Handle image response
        if images and "image" in data:
            from PIL import Image

            if "base64" in data["image"]:
                img_data = base64.b64decode(data["image"]["base64"])
                return Image.open(BytesIO(img_data)) if return_image else img_data
            elif "url" in data["image"]:
                img_response = requests.get(data["image"]["url"], timeout=30)
                img_response.raise_for_status()
                return (Image.open(BytesIO(img_response.content)) if return_image
                       else img_response.content)

        # Handle text response
        for key in ["choices", "generated_text", "output"]:
            if key == "choices" and key in data and data[key]:
                return data[key][0]["text"]
            elif key in data:
                return data[key]
        return data

    def save_image(self, image, output_path: str):
        """Save image to file."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow required. Install: pip install Pillow")

        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    def cleanup(self):
        """Cleanup resources."""
        self.stop_server()
        if self.pipeline:
            del self.pipeline, self.model, self.tokenizer
            self.pipeline = self.model = self.tokenizer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        self.cleanup()
