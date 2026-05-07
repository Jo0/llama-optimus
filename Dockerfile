# =============================================================================
# llama-optimus Dockerfile
# =============================================================================
# Single Dockerfile supporting all llama.cpp GPU backends via build argument.
#
# Usage:
#   # CUDA 13 (default development target)
#   docker build --build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-cuda13 \
#       -t llama-optimus-cuda13 .
#
#   # CUDA 12
#   docker build --build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-cuda \
#       -t llama-optimus-cuda .
#
#   # CPU-only
#   docker build --build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full \
#       -t llama-optimus-cpu .
#
#   # See docs/DOCKER_ARCHITECTURE.md for all supported variants.
# =============================================================================

ARG BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-cuda13
FROM $BASE_IMAGE

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABEL maintainer="llama-optimus"
LABEL description="llama-optimus: automatic llama.cpp performance flag optimizer"
LABEL org.opencontainers.image.source="https://github.com/BrunoArsioli/llama-optimus"

# ---------------------------------------------------------------------------
# Install Python runtime
# ---------------------------------------------------------------------------
# The base image ships compiled llama.cpp binaries (llama-bench, llama-server)
# but may not include Python.  Install Python 3, pip, and venv support.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Copy project files
# ---------------------------------------------------------------------------
# Copy dependency manifest first for better layer caching.
COPY pyproject.toml ./
COPY requirements.txt ./

# Copy source code.
COPY src/ src/
COPY optimus.py ./

# ---------------------------------------------------------------------------
# Install Python dependencies (editable mode so imports resolve correctly)
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# HuggingFace cache location (matches the default volume mount).
ENV HF_HOME=/root/.cache/huggingface

# Ensure Python output is streamed to docker logs without buffering.
ENV PYTHONUNBUFFERED=1

# Default path to llama.cpp binaries inside the container.
# The official llama.cpp Docker images place compiled binaries under
# /opt/llama.cpp/build/bin or /usr/local/bin.  This default covers the
# most common layout; override with --llama-bin if needed.
ENV LLAMA_BIN=/opt/llama.cpp/build/bin

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
ENTRYPOINT ["python3", "optimus.py"]
CMD ["--help"]
