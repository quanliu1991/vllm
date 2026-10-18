#!/bin/bash
# Baseline server: production-equivalent params + reasoning/tool parsing.
# Production (hb-serve run.sh): reasoning_parser="qwen3" (qwen3.5 map),
# tool_call_parser="qwen3_coder".
# v0.30.0: serving-side parsing needs --reasoning-parser; --reasoning-config
# only manages thinking-budget token IDs. Prefix caching DISABLED per spec.

export VLLM_CONFIGURE_LOGGING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export DISABLE_CPU_OFFLOAD=true
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE:-true}

MODEL=${MODEL:-/models/Qwen3.6-35B-A3B}
PORT=${PORT:-35005}
EXTRA_ARGS=${EXTRA_ARGS:-}

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name HengNao-v4 \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 262144 \
  --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --port "$PORT" \
  $EXTRA_ARGS
