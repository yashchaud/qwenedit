#!/bin/bash

# Entrypoint script to support both RunPod Serverless and Pods

if [ "$DEPLOYMENT_MODE" = "pod" ]; then
    echo "Starting in Pod mode (FastAPI server)..."
    exec python -u api_server.py
else
    echo "Starting in Serverless mode (RunPod handler)..."
    exec python -u handler.py
fi
