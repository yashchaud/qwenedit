#!/bin/bash

# Update system
apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Verify installation
docker --version

# Test Docker
docker ps

echo "Docker installed successfully!"
