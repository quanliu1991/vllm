#!/bin/bash
# Config-variant server launcher. Usage:
#   bash run_variant.sh <tag> [extra vllm args...]
# Common baseline settings except what the variant overrides.

export VLLM_CONFIGURE_LOGGING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export DISABLE_CPU_OFFLOAD=true

TAG=$1; shift
PORT=${PORT:-35005}

MODEL=${MODEL:-/models/Qwen3.6-35B-A3B}

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name HengNao-v4 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 262144 \
  --no-enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --port "$PORT" \
  "$@" \
  2>&1 | sed "s/^/[$TAG] /"
