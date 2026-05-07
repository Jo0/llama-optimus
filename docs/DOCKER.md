# Docker Quick Start

Run llama-optimus inside a Docker container to eliminate host-side dependencies (Python, pip, virtual environments) while leveraging pre-compiled `llama.cpp` binaries from official Docker images.

## Prerequisites

- **Docker Engine** (v20.10+)
- **NVIDIA Container Toolkit** (for CUDA variants) — [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- At least one **GGUF model file**

## Quick Start (CUDA 13)

### 1. Build the image

```bash
docker build \
    --build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-cuda13 \
    -t llama-optimus-cuda13 .
```

### 2. Run the optimizer

```bash
docker run --gpus all \
    -v /path/to/your/models:/app/models:ro \
    -v /path/to/your/output:/app/output \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    llama-optimus-cuda13 \
    --model /app/models/my-model.gguf \
    --llama-bin /opt/llama.cpp/build/bin \
    --trials 45 --metric tg
```

### 3. Quick test (5 trials, minimal)

```bash
docker run --gpus all \
    -v /path/to/your/models:/app/models:ro \
    llama-optimus-cuda13 \
    --model /app/models/my-model.gguf \
    --llama-bin /opt/llama.cpp/build/bin \
    --trials 5 --repeat 1 --no-warmup --n-tokens 20 --metric tg
```

## Supported Variants

Build the image with the desired base image for your GPU backend:

| Backend | Build Command | Run Flags |
|---------|--------------|-----------|
| **NVIDIA CUDA 13** | `--build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-cuda13` | `--gpus all` |
| **NVIDIA CUDA 12** | `--build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-cuda` | `--gpus all` |
| **AMD ROCm** | `--build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-rocm` | `--device /dev/kfd --device /dev/dri --group-add video` |
| **Vulkan** | `--build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-vulkan` | `--device /dev/dri --group-add video` |
| **Intel** | `--build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-intel` | `--device /dev/dri --group-add video` |
| **OpenVINO** | `--build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-openvino` | (none required) |
| **CPU-only** | `--build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full` | (none required) |
| **Musa** | `--build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-musa` | (check Musa docs) |
| **S390x** | `--build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-s390x` | (none required) |

### Example: Build for AMD ROCm

```bash
docker build \
    --build-arg BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:full-rocm \
    -t llama-optimus-rocm .

docker run --device /dev/kfd --device /dev/dri --group-add video \
    -v /path/to/models:/app/models:ro \
    -v /path/to/output:/app/output \
    llama-optimus-rocm \
    --model /app/models/my-model.gguf \
    --llama-bin /opt/llama.cpp/build/bin \
    --trials 45 --metric tg
```

## Using docker-compose

The included [`docker-compose.yml`](../docker-compose.yml) provides a convenient way to run the optimizer without typing long `docker run` commands.

### 1. Configure paths

Edit `docker-compose.yml` and update the volume mounts to match your paths:

```yaml
volumes:
  - ./models:/models:ro          # Your models directory
  - ./output:/app/output         # Output directory
  - ~/.cache/huggingface:/root/.cache/huggingface  # HF cache
```

### 2. Run

```bash
# Full optimization
docker compose run optimus \
    --model /models/my-model.gguf \
    --llama-bin /opt/llama.cpp/build/bin \
    --trials 45 --metric tg

# Quick test
docker compose run optimus \
    --model /models/my-model.gguf \
    --llama-bin /opt/llama.cpp/build/bin \
    --trials 5 --repeat 1 --no-warmup --n-tokens 20 --metric tg
```

## Build All Variants

Use the helper script to build images for all GPU backends:

```bash
# Build all variants
./scripts/build_docker.sh

# Build a single variant
./scripts/build_docker.sh cuda13

# Build with custom prefix
PREFIX=myrepo ./scripts/build_docker.sh
```

## Volume Mount Reference

| Container Path | Host Mount | Purpose |
|---------------|------------|---------|
| `/app/models` | Your models directory | GGUF model files (read-only) |
| `/app/output` | Your output directory | Results, logs, config JSON |
| `/root/.cache/huggingface` | `~/.cache/huggingface` | HuggingFace download cache |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_BIN` | `/opt/llama.cpp/build/bin` | Path to llama.cpp binaries inside container |
| `MODEL_PATH` | (unset) | Path to model file (alternative to `--model`) |
| `HF_HOME` | `/root/.cache/huggingface` | HuggingFace cache directory |
| `PYTHONUNBUFFERED` | `1` | Stream Python output to docker logs |

## Troubleshooting

### `llama-bench not found`

The binary path may differ from the default. Check the actual location inside the container:

```bash
docker run --rm llama-optimus-cuda13 find / -name "llama-bench" -type f 2>/dev/null
```

Then pass the correct path with `--llama-bin`:

```bash
docker run --gpus all \
    -v /path/to/models:/app/models:ro \
    llama-optimus-cuda13 \
    --model /app/models/my-model.gguf \
    --llama-bin /actual/path/to/llama-bench-parent-dir \
    --trials 5 --metric tg
```

### NVIDIA: `CUDA driver version is insufficient`

Ensure the NVIDIA Container Toolkit is installed and the Docker daemon is restarted:

```bash
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify with:

```bash
docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi
```

### AMD ROCm: `/dev/kfd: permission denied`

Add the `video` group and mount the required devices:

```bash
docker run --device /dev/kfd --device /dev/dri --group-add video --group-add render \
    llama-optimus-rocm \
    --model /app/models/my-model.gguf \
    --llama-bin /opt/llama.cpp/build/bin \
    --trials 5 --metric tg
```

### Out of Memory (OOM)

Large models may need more container memory. Increase the limit:

```bash
docker run --gpus all --memory=32g \
    -v /path/to/models:/app/models:ro \
    llama-optimus-cuda13 \
    --model /app/models/my-model.gguf \
    --llama-bin /opt/llama.cpp/build/bin \
    --trials 45 --metric tg
```

## Architecture Details

For a deeper dive into the design decisions, volume strategy, and multi-variant build approach, see [`DOCKER_ARCHITECTURE.md`](DOCKER_ARCHITECTURE.md).
