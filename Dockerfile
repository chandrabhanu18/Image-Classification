# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create directories first
RUN mkdir -p data/train/cat data/train/dog data/val/cat data/val/dog data/test/cat data/test/dog
RUN mkdir -p models visualizations

# Copy project files
COPY . .

# Install CPU-only PyTorch to avoid CUDA downloads in slim image
RUN pip install --default-timeout=1000 --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu

# Install remaining dependencies from PyPI
RUN pip install --default-timeout=1000 --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    Pillow==10.0.0 \
    opencv-python==4.8.0.74 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    scikit-learn==1.3.0 \
    tqdm==4.66.1 \
    PyYAML==6.0.1 \
    requests==2.32.5

# Expose port for Jupyter (optional)
EXPOSE 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command: show help
CMD ["python", "--version"]
