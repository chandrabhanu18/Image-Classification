# Docker Deployment Guide

## Overview

This guide explains how to run the Transfer Learning Image Classifier in Docker containers.

## Prerequisites

- Docker installed ([Download](https://www.docker.com/products/docker-desktop))
- Docker Compose (included with Docker Desktop)
- ~3GB disk space for image

## Quick Start

### 1. Build the Docker Image

```bash
cd "/d/Chandra's Work/GPP/week-8"
docker build -t transfer-learning:latest .
```

**Output:**
```
Successfully built abc123def456
Successfully tagged transfer-learning:latest
```

### 2. Run Training in Container

```bash
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           -v $(pwd)/visualizations:/app/visualizations \
           transfer-learning:latest \
           python train.py --config config.yaml
```

**What this does:**
- Mounts local `data/`, `models/`, `visualizations/` folders
- Runs training inside container
- Saves results to your local machine

### 3. Run Inference in Container

```bash
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           -v $(pwd)/visualizations:/app/visualizations \
           transfer-learning:latest \
           python predict.py --checkpoint models/resnet50_finetuned.pth \
                            --image data/test/cat/cat.0.jpg --gradcam \
                            --output visualizations/gradcam_docker.png
```

---

## Using Docker Compose (Easier)

### 1. Build with Compose

```bash
docker-compose build
```

### 2. Run Training

```bash
docker-compose run transfer-learning
```

### 3. Run Jupyter Notebook

```bash
docker-compose up jupyter
```

Then open: `http://localhost:8888`

### 4. Run Inference

```bash
docker-compose run transfer-learning python predict.py \
  --checkpoint models/resnet50_finetuned.pth \
  --image data/test/cat/cat.0.jpg --gradcam
```

### 5. Stop All Services

```bash
docker-compose down
```

---

## Docker Commands Reference

### Image Management

```bash
# Build image
docker build -t transfer-learning:latest .

# List images
docker images

# Remove image
docker rmi transfer-learning:latest

# View image history
docker history transfer-learning:latest
```

### Container Management

```bash
# Run container
docker run -it transfer-learning:latest bash

# List running containers
docker ps

# List all containers
docker ps -a

# Stop container
docker stop <container_id>

# Remove container
docker rm <container_id>

# View logs
docker logs <container_id>

# Enter running container
docker exec -it <container_id> bash
```

### Volume Management

```bash
# Inspect volume
docker volume ls

# Remove unused volumes
docker volume prune
```

---

## Dockerfile Explanation

```dockerfile
FROM python:3.10-slim
# Lightweight Python 3.10 base image

WORKDIR /app
# Set working directory in container

COPY requirements.txt .
# Copy dependencies file (cached layer)

RUN pip install --no-cache-dir -r requirements.txt
# Install Python packages

COPY . .
# Copy entire project

CMD ["python", "train.py", "--config", "config.yaml"]
# Default command when container runs
```

---

## Common Use Cases

### Case 1: Train with Custom Data

```bash
# First, copy your data to data/ folder locally
# Then:
docker-compose run transfer-learning python train.py --config config.yaml

# Results saved in ./models/ and ./visualizations/
```

### Case 2: Batch Prediction

```bash
# Create a script: predict_batch.py
for image in data/test/cat/*.jpg; do
  docker run -v $(pwd):/app transfer-learning:latest \
    python predict.py --checkpoint models/resnet50_finetuned.pth \
                     --image "$image"
done
```

### Case 3: Interactive Development

```bash
# Open bash shell inside container
docker run -it -v $(pwd):/app transfer-learning:latest bash

# Inside container, you can:
python train.py --config config.yaml
python predict.py --checkpoint models/resnet50_finetuned.pth --image data/test/cat/cat.0.jpg
jupyter notebook --ip=0.0.0.0
```

### Case 4: Run with GPU Support

```bash
# Install nvidia-docker first
nvidia-docker run --gpus all -v $(pwd):/app transfer-learning:latest \
  python train.py --config config.yaml
```

---

## Troubleshooting

### Issue: "Docker daemon is not running"
**Solution:** Start Docker Desktop or Docker service

### Issue: "permission denied while trying to connect to Docker daemon"
**Solution (Linux/Mac):**
```bash
sudo usermod -aG docker $USER
# Restart terminal
```

### Issue: "Out of memory during build"
**Solution:**
```bash
# Allocate more memory in Docker Desktop settings
# Or reduce layer complexity in Dockerfile
```

### Issue: "Cannot find data files in container"
**Solution:** Ensure volumes are mounted correctly
```bash
# Check mount with:
docker run -it transfer-learning:latest ls -la /app/data
```

### Issue: "Port 8888 already in use"
**Solution:**
```bash
# Map to different port:
docker run -p 8889:8888 transfer-learning:latest jupyter notebook --ip=0.0.0.0
```

---

## File Sizes

- **Base image (python:3.10-slim):** ~150 MB
- **After dependencies:** ~2.5 GB
- **Final image:** ~2.6 GB

---

## Production Deployment

For production, create a more optimized Dockerfile:

```dockerfile
# Multi-stage build for production
FROM python:3.10-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache /wheels/*
COPY . .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8888
CMD ["python", "train.py", "--config", "config.yaml"]
```

---

## Summary

✅ **Docker Setup Complete:**
- `Dockerfile` - Container image definition
- `docker-compose.yml` - Multi-container orchestration
- `.dockerignore` - Exclude unnecessary files
- `DOCKER.md` - This guide

**Test it:**
```bash
docker build -t transfer-learning:latest .
docker run -v $(pwd):/app transfer-learning:latest python train.py --config config.yaml
```

**Your project is now containerized and deployment-ready! 🐳**
