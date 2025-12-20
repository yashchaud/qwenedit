# Self-Hosted GitHub Actions Runner Setup Guide

This guide will help you set up a self-hosted GitHub Actions runner on a RunPod CPU instance to build Docker images for the Qwen-Image-Layered inference server.

## Why Self-Hosted Runner?

- GitHub Actions runners have limited disk space (~14GB free)
- Our Docker build requires ~50-80GB for dependencies and model downloads
- Self-hosted runner on RunPod gives us full control over resources

## RunPod Instance Requirements

### Recommended Specs
- **CPU**: 16 vCPUs
- **RAM**: 32GB
- **Disk**: 100GB+ NVMe SSD
- **Cost**: ~$0.20-0.40/hour

### Budget Option (Minimum)
- **CPU**: 8 vCPUs
- **RAM**: 16GB
- **Disk**: 80GB SSD
- **Cost**: ~$0.15-0.25/hour

## Step-by-Step Setup

### 1. Rent RunPod CPU Instance

1. Go to [RunPod Pods](https://www.runpod.io/console/pods)
2. Click **"CPU"** tab
3. Select a template:
   - **Recommended**: "RunPod Pytorch" or "Ubuntu 22.04"
4. Configure instance:
   - **Container Disk**: 100GB minimum
   - **Expose HTTP Ports**: 22 (SSH)
   - **Volume**: Optional (not needed for this use case)
5. Click **"Deploy On-Demand"**
6. Wait for instance to start (~30 seconds)
7. Note the **SSH connection string** (looks like: `ssh root@XX.XXX.XX.XXX -p XXXXX -i ~/.ssh/id_ed25519`)

### 2. Connect to Your Instance

```bash
# Use the SSH command provided by RunPod
ssh root@XX.XXX.XX.XXX -p XXXXX -i ~/.ssh/id_ed25519

# Or if using password authentication
ssh root@XX.XXX.XX.XXX -p XXXXX
```

### 3. Install Docker (if not pre-installed)

```bash
# Check if Docker is already installed
docker --version

# If not installed, run:
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Verify installation
docker --version
docker ps
```

### 4. Install Required Dependencies

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install essential tools
apt-get install -y git curl wget jq

# Verify installations
git --version
curl --version
```

### 5. Set Up GitHub Actions Runner

#### Get Registration Token

1. Go to your repository: `https://github.com/yashchaud/qwenedit`
2. Click **Settings** → **Actions** → **Runners**
3. Click **"New self-hosted runner"**
4. Select **Linux** and **x64**
5. Copy the registration token (starts with `A...`)

#### Install and Configure Runner

```bash
# Create directory for runner
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download latest runner (check GitHub for latest version)
curl -o actions-runner-linux-x64-2.319.1.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.319.1/actions-runner-linux-x64-2.319.1.tar.gz

# Extract
tar xzf ./actions-runner-linux-x64-2.319.1.tar.gz

# Configure runner (replace YOUR_TOKEN with the token from GitHub)
./config.sh \
  --url https://github.com/yashchaud/qwenedit \
  --token YOUR_TOKEN \
  --name runpod-builder \
  --labels runpod,docker,linux

# When prompted:
# - Runner group: Press Enter (default)
# - Additional labels: Press Enter (we already set them)
# - Work folder: Press Enter (default: _work)
```

### 6. Run the Runner

#### Option A: Run Interactively (for testing)

```bash
# Run in foreground (good for testing)
./run.sh
```

Press `Ctrl+C` to stop when done testing.

#### Option B: Run as Service (recommended for production)

```bash
# Install as systemd service
sudo ./svc.sh install

# Start the service
sudo ./svc.sh start

# Check status
sudo ./svc.sh status

# View logs
journalctl -u actions.runner.* -f

# To stop the service
sudo ./svc.sh stop
```

### 7. Verify Runner is Connected

1. Go to: `https://github.com/yashchaud/qwenedit/settings/actions/runners`
2. You should see your runner listed with a green "Idle" status
3. Runner name: `runpod-builder`
4. Labels: `self-hosted`, `Linux`, `X64`, `runpod`, `docker`, `linux`

## Testing the Setup

### Trigger a Build

1. **Manual trigger**:
   - Go to: `https://github.com/yashchaud/qwenedit/actions`
   - Click "Build and Push Docker Image to GHCR"
   - Click "Run workflow"
   - Select `main` branch
   - Click "Run workflow"

2. **Or push a commit**:
   ```bash
   git commit --allow-empty -m "Test self-hosted runner"
   git push origin main
   ```

### Monitor the Build

```bash
# On your RunPod instance, watch Docker
watch docker ps

# Monitor disk usage
watch df -h

# View runner logs
cd ~/actions-runner
tail -f _diag/Runner_*.log
```

## Cost Management

### On-Demand Usage (Pay only when building)

1. **Before build**: Start RunPod instance + configure runner (~5 min setup)
2. **Build**: Let GitHub Actions run (~15-20 min build)
3. **After build**: Terminate instance
4. **Cost per build**: ~$0.10-0.15

### Keep Running (If you build frequently)

- **Cost**: ~$0.30/hour × 24 hours = ~$7.20/day
- **Good for**: Active development with frequent builds
- **Remember to**: Terminate when not needed for extended periods

## Troubleshooting

### Runner Not Showing Up

```bash
# Check if runner service is running
sudo ./svc.sh status

# Restart runner
sudo ./svc.sh stop
sudo ./svc.sh start

# Check logs
cd ~/actions-runner
cat _diag/Runner_*.log
```

### Build Fails with "No Space Left"

```bash
# Check disk usage
df -h

# Clean up Docker
docker system prune -af --volumes

# Remove old images
docker image prune -af

# If still no space, increase container disk in RunPod settings
```

### Docker Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Re-login or run
newgrp docker

# Verify
docker ps
```

### Runner Disconnects

```bash
# Check runner logs
cd ~/actions-runner
tail -f _diag/Runner_*.log

# Restart runner service
sudo ./svc.sh restart
```

## Maintenance

### Update Runner

```bash
cd ~/actions-runner

# Stop runner
sudo ./svc.sh stop

# Download latest version
curl -o actions-runner-linux-x64-2.XXX.X.tar.gz -L \
  https://github.com/actions/runner/releases/download/vX.XXX.X/actions-runner-linux-x64-2.XXX.X.tar.gz

# Extract
tar xzf ./actions-runner-linux-x64-2.XXX.X.tar.gz

# Start runner
sudo ./svc.sh start
```

### Clean Up Old Builds

```bash
# Remove old Docker images and containers
docker system prune -af --volumes

# Remove old build artifacts
cd ~/actions-runner/_work
rm -rf qwenedit/qwenedit/*

# Check disk space
df -h
```

## Security Best Practices

1. **Use SSH Keys**: Don't use password authentication
2. **Firewall**: Only expose port 22 (SSH)
3. **Regular Updates**: Keep runner and Docker updated
4. **Monitor Usage**: Check RunPod dashboard for unexpected costs
5. **Terminate When Idle**: Stop instance if not building for >24 hours

## Quick Reference

### Useful Commands

```bash
# Check runner status
sudo ./svc.sh status

# View runner logs
journalctl -u actions.runner.* -f

# Check disk space
df -h

# Clean Docker
docker system prune -af

# Check Docker images
docker images

# View running containers
docker ps

# Monitor resources
htop
```

### Important URLs

- **GitHub Runners**: https://github.com/yashchaud/qwenedit/settings/actions/runners
- **GitHub Actions**: https://github.com/yashchaud/qwenedit/actions
- **RunPod Console**: https://www.runpod.io/console/pods
- **GitHub Packages**: https://github.com/yashchaud?tab=packages

## Summary

**Total Setup Time**: ~10-15 minutes
**Build Time**: ~15-20 minutes
**Cost per Build**: ~$0.10-0.15 (on-demand)
**Disk Required**: 80-100GB
**RAM Required**: 16-32GB

Once set up, your runner will automatically build Docker images whenever you push to the main branch or create a version tag, then push them to GitHub Container Registry where RunPod serverless can access them.
