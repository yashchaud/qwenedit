# RunPod Serverless Dockerfile for Qwen-Image-Layered
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git && \
    pip install --no-cache-dir runpod

# Note: Model weights (~40GB) are NOT pre-downloaded in the image due to size constraints
# They will be downloaded on first inference and cached in RunPod's network volume
# This results in a slower first cold start (~2-3 minutes) but much smaller image size

# Copy application code
COPY handler.py .
COPY utils/ ./utils/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Note: Health check removed to prevent build issues
# RunPod will handle health checks automatically

# Start handler
CMD ["python", "-u", "handler.py"]
