#!/bin/bash
# =============================================================================
# build_docker.sh
# =============================================================================
# Build llama-optimus Docker images for all supported GPU backends.
#
# Usage:
#   # Build all variants
#   ./scripts/build_docker.sh
#
#   # Build a single variant
#   ./scripts/build_docker.sh cuda13
#
#   # Build with a custom image prefix
#   PREFIX=myrepo ./scripts/build_docker.sh
# =============================================================================

set -euo pipefail

# Image prefix (override via environment)
PREFIX="${PREFIX:-llama-optimus}"

# Define variants: "base_image_tag:short_name"
VARIANTS=(
    "full:cpu"
    "full-cuda:cuda"
    "full-cuda13:cuda13"
    "full-rocm:rocm"
    "full-musa:musa"
    "full-intel:intel"
    "full-vulkan:vulkan"
    "full-openvino:openvino"
    "full-s390x:s390x"
)

BASE_REGISTRY="ghcr.io/ggml-org/llama.cpp"

build_variant() {
    local base_tag="$1"
    local short_name="$2"
    local image_name="${PREFIX}-${short_name}"

    echo "============================================"
    echo "Building ${image_name}"
    echo "  Base image: ${BASE_REGISTRY}:${base_tag}"
    echo "============================================"

    docker build \
        --build-arg BASE_IMAGE="${BASE_REGISTRY}:${base_tag}" \
        -t "${image_name}" \
        .

    echo ""
    echo "  Built: ${image_name}"
    echo ""
}

# If a specific variant is requested
if [ $# -gt 0 ]; then
    requested="$1"
    for variant in "${VARIANTS[@]}"; do
        IFS=':' read -r base_tag short_name <<< "$variant"
        if [ "$short_name" = "$requested" ]; then
            build_variant "$base_tag" "$short_name"
            exit 0
        fi
    done
    echo "Unknown variant: ${requested}"
    echo "Available variants: ${VARIANTS[*]}"
    exit 1
fi

# Build all variants
echo "Building all llama-optimus Docker variants..."
echo ""

for variant in "${VARIANTS[@]}"; do
    IFS=':' read -r base_tag short_name <<< "$variant"
    build_variant "$base_tag" "$short_name"
done

echo "============================================"
echo "All variants built successfully."
echo "============================================"
echo ""
echo "Images:"
for variant in "${VARIANTS[@]}"; do
    IFS=':' read -r _ short_name <<< "$variant"
    echo "  ${PREFIX}-${short_name}"
done
