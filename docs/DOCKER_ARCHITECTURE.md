# Docker Architecture for llama-optimus

## Overview

This document outlines the approach for containerizing llama-optimus using official llama.cpp Docker images as base images. The goal is to reduce host setup friction while keeping the user in control of model paths, outputs, and the `llama-bench` binary that is already baked into the base image.

## Why Extend the Official llama.cpp Images?

The official images from `ghcr.io/ggml-org/llama.cpp` already contain:

- Compiled `llama-bench` and `llama-server` binaries
- CUDA / ROCm / Vulkan / OpenVINO runtime libraries
- System dependencies (libc, libstdc++, etc.)

By extending one of these images, we avoid:
- Building llama.cpp from source inside the container
- Managing CMake, compilers, and GPU SDK toolchains
- Inconsistent binary versions between `llama-bench` and the runtime libraries

## Supported Base Images

| Variant | Base Image | GPU Backend |
|---------|-----------|-------------|
| CUDA 12 (default dev target) | `ghcr.io/ggml-org/llama.cpp:full-cuda` | NVIDIA CUDA 12.x |
| CUDA 13 | `ghcr.io/ggml-org/llama.cpp:full-cuda13` | NVIDIA CUDA 13.x |
| ROCm | `ghcr.io/ggml-org/llama.cpp:full-rocm` | AMD ROCm |
| Musa | `ghcr.io/ggml-org/llama.cpp:full-musa` | Huawei Ascend |
| Intel | `ghcr.io/ggml-org/llama.cpp:full-intel` | Intel GPU / OpenVINO |
| Vulkan | `ghcr.io/ggml-org/llama.cpp:full-vulkan` | Any Vulkan-capable GPU |
| OpenVINO | `ghcr.io/ggml-org/llama.cpp:full-openvino` | Intel CPU / GPU via OpenVINO |
| Generic (no GPU) | `ghcr.io/ggml-org/llama.cpp:full` | CPU only |
| S390x | `ghcr.io/ggml-org/llama.cpp:full-s390x` | IBM Z mainframe |

## Architecture Diagram

```mermaid
flowchart TB
    subgraph Host
        ModelsDir[Models Directory]
        HFCache[HuggingFace Cache]
        OutputDir[Output Directory]
        DockerEngine[Docker Engine]
    end

    subgraph Container[llama.optimus Container]
        BaseImg[llama.cpp Base Image<br/>llama-bench, llama-server]
        PythonLayer[Python 3.x + venv]
        AppCode[llama.optimus Source]
        Entrypoint[Entry Point Script]
    end

    ModelsDir -. volume mount .-> ModelsDir2[/app/models]
    HFCache -. volume mount .-> HFCache2[/root/.cache/huggingface]
    OutputDir -. volume mount .-> OutputDir2[/app/output]

    DockerEngine --> Container
    BaseImg --> PythonLayer
    PythonLayer --> AppCode
    AppCode --> Entrypoint

    Entrypoint -->|calls| llama-bench[llama-bench binary]
    Entrypoint -->|reads| ModelsDir2
    Entrypoint -->|writes| OutputDir2
```

## Directory Layout Inside Container

```
/app/
  src/                    # llama.optimus source code
  logs/                   # optimization logs (volume mounted)
  best_phase1_config.json # Phase 1 output (volume mounted)
  output/                 # results, CSV exports (volume mounted)
  models/                 # model directory (volume mounted)
  .cache/huggingface/     # HF cache (volume mounted)
```

## Volume Mount Strategy

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./models` (user specified) | `/app/models` | Model files (GGUF) |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | HuggingFace download cache |
| `./output` (user specified) | `/app/output` | Results, logs, config JSON |

## GPU Passthrough

### NVIDIA (CUDA)

Use the NVIDIA Container Toolkit (`nvidia-container-toolkit`). The `--gpus all` flag handles device mapping, driver mounting, and CUDA library injection.

```bash
docker run --gpus all -v ./models:/app/models -v ./output:/app/output llama-optimus-cuda13
```

### AMD (ROCm)

Use `--device /dev/kfd --device /dev/dri` and set appropriate environment variables.

```bash
docker run --device /dev/kfd --device /dev/dri --group-add video -v ./models:/app/models llama-optimus-rocm
```

### Vulkan / Generic

Vulkan requires `/dev/dri` access.

```bash
docker run --device /dev/dri --group-add video -v ./models:/app/models llama-optimus-vulkan
```

## Build Approach

The user builds the image locally from a `Dockerfile` in this repository. The Dockerfile:

1. Takes the base image as a build argument (default: `full-cuda13`)
2. Installs Python and pip
3. Copies project source code
4. Installs Python dependencies (`optuna`, `pandas`, `gguf`)
5. Sets the entry point to `optimus.py` (or the CLI entry point)

## Dockerfile Design

```dockerfile
# Syntax: docker build --build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-cuda13 -t llama-optimus-cuda13 .

ARG BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-cuda13
FROM $BASE_IMAGE

# Install Python and pip (the base image may already have Python; we ensure it)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml setup.py* ./
COPY src/ src/
COPY optimus.py ./
COPY requirements.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir -e .

# Set default environment
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# Default entrypoint
ENTRYPOINT ["python3", "optimus.py"]
CMD ["--help"]
```

## Build and Run Commands

### Build

```bash
# For CUDA 13 (default development target)
docker build --build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-cuda13 -t llama-optimus-cuda13 .

# For CUDA 12
docker build --build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-cuda -t llama-optimus-cuda .

# For CPU-only
docker build --build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full -t llama-optimus-cpu .
```

### Run

```bash
# Basic run with CUDA 13
docker run --gpus all \
    -v /path/to/models:/app/models \
    -v /path/to/output:/app/output \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    llama-optimus-cuda13 \
    --model /app/models/my-model.gguf \
    --llama-bin /opt/llama.cpp/build/bin \
    --trials 45 --metric tg

# Quick test
docker run --gpus all \
    -v /path/to/models:/app/models \
    llama-optimus-cuda13 \
    --model /app/models/my-model.gguf \
    --llama-bin /opt/llama.cpp/build/bin \
    --trials 5 --repeat 1 --no-warmup --n-tokens 20 --metric tg
```

## Llama-bench Binary Location

The official llama.cpp images place compiled binaries in a known location. We need to determine the exact path. Based on the llama.cpp Docker build process, binaries are typically at `/opt/llama.cpp/build/bin/` or `/usr/local/bin/`. The container entry point or documentation should specify the correct `--llama-bin` path.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_BIN` | `/opt/llama.cpp/build/bin` | Path to llama.cpp binaries inside container |
| `MODEL_PATH` | (unset) | Path to model file (overrides `--model`) |
| `HF_HOME` | `/root/.cache/huggingface` | HuggingFace cache directory |
| `PYTHONUNBUFFERED` | `1` | Ensures Python output is streamed to docker logs |

## Multi-variant Build Script

A helper script (`scripts/build_docker.sh`) could automate building all variants:

```bash
#!/bin/bash
VARIANTS=(
    "full:cpu"
    "full-cuda:cuda"
    "full-cuda13:cuda13"
    "full-rocm:rocm"
    "full-musa:musa"
    "full-intel:intel"
    "full-vulkan:vulkan"
    "full-openvino:openvino"
)

for variant in "${VARIANTS[@]}"; do
    IFS=':' read -r base suffix <<< "$variant"
    echo "Building llama-optimus-${suffix} ..."
    docker build \
        --build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:${base} \
        -t llama-optimus-${suffix} .
done
```

## Key Design Decisions

1. **User builds the image** - No public registry publishing. The repo contains the Dockerfile and build instructions.

2. **Base image as build argument** - Single Dockerfile serves all GPU backends. The user chooses the target backend at build time.

3. **Volume mounts for models and output** - Models are typically large (several GB). Volume mounting avoids copying them into the image. Output directory is mounted so results persist after container stops.

4. **HF cache volume mount** - Prevents re-downloading models on every run. The host `~/.cache/huggingface` is shared with the container.

5. **Python installed in container** - The base image focuses on C++ binaries. Python and its dependencies are layered on top.

6. **Entry point is `optimus.py`** - The container runs the optimizer directly. All CLI arguments are passed through `CMD`.

## Files to Create

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-variant Docker image definition |
| `.dockerignore` | Exclude host artifacts from build context |
| `docker-compose.yml` | Optional convenience for local development |
| `scripts/build_docker.sh` | Helper script to build all variants |
| `docs/DOCKER.md` | User-facing Docker documentation |

## Potential Issues and Mitigations

| Issue | Mitigation |
|-------|-----------|
| Base image Python version unknown | Install Python explicitly in Dockerfile |
| Binary path differs across base image versions | Detect binary location in entrypoint or document known paths |
| Large build context | Use `.dockerignore` to exclude `.git`, `.venv`, logs, models |
| NVIDIA driver version mismatch | User must have `nvidia-container-toolkit` installed on host |
| Memory limits for large models | Document `--memory` flag for `docker run` |
