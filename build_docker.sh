#!/bin/bash
set -e

IMAGE_NAME="${IMAGE_NAME:-docker.das-security.cn/hb/hb-serve-chat-npu}"
IMAGE_TAG="${IMAGE_TAG:-R26C10-v4.1.0-o1}"
BUILD_CONTEXT="${BUILD_CONTEXT:-.}"

echo "Building NPU Docker image..."
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Dockerfile: Dockerfile.npu"
echo "Build context: ${BUILD_CONTEXT}"

docker build -f Dockerfile.npu -t "${IMAGE_NAME}:${IMAGE_TAG}" "${BUILD_CONTEXT}"

echo "Build complete: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Usage examples:"
echo "  # Start both LLM and embedding (default)"
echo "  docker run -d --name vllm-npu ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "  # Start only LLM service"
echo "  docker run -d --name vllm-npu -e START_LLM=true -e START_EMBEDDING=false ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "  # Start only embedding service"
echo "  docker run -d --name vllm-npu -e START_LLM=false -e START_EMBEDDING=true ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "  # Start both services explicitly"
echo "  docker run -d --name vllm-npu -e START_LLM=true -e START_EMBEDDING=true ${IMAGE_NAME}:${IMAGE_TAG}"