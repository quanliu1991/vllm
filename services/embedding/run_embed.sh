#!/bin/bash

if [ -n "$EMBED_HOST" ]; then
    embed_host=$EMBED_HOST
else
    embed_host="0.0.0.0"
fi

if [ -n "$EMBED_PORT" ]; then
    embed_port=$EMBED_PORT
else
    embed_port=34002
fi

if [ -n "$EMBED_MODEL" ]; then
    embed_model=$EMBED_MODEL
else
    embed_model="/home/models/bge-large-zh-v1.5"
fi

if [ -n "$EMBED_BATCH_SIZE" ]; then
    embed_batch_size=$EMBED_BATCH_SIZE
else
    embed_batch_size=16
fi

# Pass through environment variables used by server.py
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"
export ALLOW_CONSUL="${ALLOW_CONSUL:-false}"
export CONSUL_HOST="${CONSUL_HOST:-localhost}"
export CONSUL_PORT="${CONSUL_PORT:-8500}"
export CONSUL_TOKEN="${CONSUL_TOKEN:-}"
export SSL_CERTIFILE="${SSL_CERTIFILE:-/home/models/configs/certificate.crt}"
export SSL_KEYFILE="${SSL_KEYFILE:-/home/models/configs/private.key}"

cd /workspace/services/embedding
exec python3 server.py \
    --host "$embed_host" \
    --port "$embed_port" \
    --model "$embed_model" \
    --batch-size "$embed_batch_size"
