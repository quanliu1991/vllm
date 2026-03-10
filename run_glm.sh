#!/bin/bash

if [ -n "$HOST" ]; then
    host=$HOST
else
    host=0.0.0.0
fi

if [ -n "$PORT" ]; then
    port=$PORT
else
    port=30011
fi

if [ -n "$MODEL" ]; then
    model=$MODEL
else
    model="/workspace/llm_models_one_piece"
fi

if [ -n "$ALLOW_LORA" ]; then
    enable_lora=$ALLOW_LORA
else
    enable_lora=
fi

if [ -n "$LORA_MODEL" ]; then
    lora_model=$LORA_MODEL
else
    lora_model="/workspace/llm_models_one_piece"
fi

if [ -n "$ALLOW_CONSUL" ]; then
    allow_consul=$ALLOW_CONSUL
else
    allow_consul=
fi

if [ -n "$CONSUL_HOST" ]; then
    consul_host=$CONSUL_HOST
else
    consul_host="localhost"
fi

if [ -n "$CONSUL_PORT" ]; then
    consul_port=$CONSUL_PORT
else
    consul_port="8500"
fi

if [ -n "$CONSUL_TOKEN" ]; then
    consul_token=$CONSUL_TOKEN
else
    consul_token=
fi

if [ -n "$SERVED_MODEL_NAME" ]; then
    served_model_name=$SERVED_MODEL_NAME
else
    served_model_name=
fi

if [ -n "$VERSION" ]; then
    version=$VERSION
else
    version=
fi

if [ -n "$GPU_MEMORY_UTILIZATION" ]; then
    gpu_memory_utilization=$GPU_MEMORY_UTILIZATION
else
    gpu_memory_utilization=
fi

if [ -n "$TENSOR_PARALLEL_SIZE" ]; then
    tensor_parallel_size=$TENSOR_PARALLEL_SIZE
else
    tensor_parallel_size=
fi

if [ -n "$QUANTIZATION" ]; then
    quantization=$QUANTIZATION
else
    quantization=
fi

if [ -n "$MAX_MODEL_LEN" ]; then
    max_model_len=$MAX_MODEL_LEN
else
    max_model_len=
fi

if [ -n "$KV_CACHE_DTYPE" ]; then
    kv_cache_dtype=$KV_CACHE_DTYPE
else
    kv_cache_dtype=
fi

if [ -n "$ENFORCE_EAGER" ]; then
    enforce_eager=$ENFORCE_EAGER
else
    enforce_eager=
fi

if [ -n "$MAX_LORAS" ]; then
    max_loras=$MAX_LORAS
else
    max_loras=
fi

if [ -n "$TOKEN_POOL_SIZE" ]; then
    token_pool_size=$TOKEN_POOL_SIZE
else
    token_pool_size=
fi


if [ -n "$ENABLE_PREFIX_CACHING" ]; then
    enable_prefix_caching=" "
else
    enable_prefix_caching=
fi

if [ -n "$DISABLE_CUSTOM_ALL_REDUCE" ]; then
    disable_custom_all_reduce=" "
else
    disable_custom_all_reduce=
fi

if [ -n "$SWAP_SPACE" ]; then
    swap_space=$SWAP_SPACE
else
    swap_space=0
fi
if [ -n "$MAX_LOGPROBS" ]; then
    max_logprobs=$MAX_LOGPROBS
else
    max_logprobs=
fi


if [ -n "$USE_PRIORITY" ]; then
    use_priority=$USE_PRIORITY
else
    use_priority=
fi
if [ -n "$LOW_PRIORITY_FREQUENCY" ]; then
    low_priority_frequency=$LOW_PRIORITY_FREQUENCY
else
    low_priority_frequency=
fi
if [ -n "$LOW_PRIORITY_TOKENS_LIMIT" ]; then
    low_priority_tokens_limit=$LOW_PRIORITY_TOKENS_LIMIT
else
    low_priority_tokens_limit=
fi

if [ -n "$SSL_KEYFILE" ]; then
    ssl_keyfile=$SSL_KEYFILE
else
    ssl_keyfile=
fi

if [ -n "$SSL_CERTIFILE" ]; then
    ssl_certifile=$SSL_CERTIFILE
else
    ssl_certifile=
fi

if [ -n "$SSL_CA_CERTS" ]; then
    ssl_ca_certs=$SSL_CA_CERTS
else
    ssl_ca_certs=
fi

if [ -n "$SSL_CERT_REQS" ]; then
    ssl_cert_reqs=$SSL_CERT_REQS
else
    ssl_cert_reqs=
fi

if [ -n "$USE_V2_BLOCK_MANAGER" ]; then
    use_v2_block_manager=$USE_V2_BLOCK_MANAGER
else
    use_v2_block_manager=
fi

if [ -n "$NUM_LOOKAHEAD_SLOTS" ]; then
    num_lookahead_slots=$NUM_LOOKAHEAD_SLOTS
else
    num_lookahead_slots=
fi

if [ -n "$DTYPE" ]; then
    dtype=$DTYPE
else
    dtype=
fi

if [ -n "$API_KEY" ]; then
    api_key=$API_KEY
else
    api_key="hb-serve-3h4kJ92slfA9wqE7ZmQxLNb2oDqKlYcR9UptHhGz1vLc0pQx"
fi

if [ -n "$OTLP_TRACES_ENDPOINT" ]; then
    otlp_traces_endpoint=$OTLP_TRACES_ENDPOINT
else
    otlp_traces_endpoint=
fi

if [ -n "$ENABLE_TOOLS" ]; then
    enable_auto_tool_choice=$ENABLE_TOOLS
else
    enable_auto_tool_choice=" "
fi

if [ -n "$TOOL_CALL_PARSER" ]; then
    tool_call_parser=$TOOL_CALL_PARSER
else
    tool_call_parser="hermes"
fi



declare -A params=(
  ["--host"]="$host"
  ["--port"]="$port"
  ["--model"]="$model"
  ["--enable-lora"]="$enable_lora"
  ["--lora-model"]="$lora_model"
  ["--gpu-memory-utilization"]="$gpu_memory_utilization"
  ["--tensor-parallel-size"]="$tensor_parallel_size"
  ["--quantization"]="$quantization"
  ["--max-model-len"]="$max_model_len"
  ["--kv-cache-dtype"]="$kv_cache_dtype"
  ["--enforce-eager"]="$enforce_eager"
  ["--token-pool-size"]="$token_pool_size"
  ["--allow-consul"]="$allow_consul"
  ["--consul-host"]="$consul_host"
  ["--consul-port"]="$consul_port"
  ["--consul-token"]="$consul_token"
  ["--version"]="$version"
  ["--served-model"]="$served_model_name"
  ["--enable-prefix-caching"]="$enable_prefix_caching"
  ["--disable-custom-all-reduce"]="$disable_custom_all_reduce"
  ["--max-loras"]="$max_loras"
  ["--swap-space"]="$swap_space"
  ["--max-logprobs"]="$max_logprobs"
  ["--use-priority"]="$use_priority"
  ["--low-priority-frequency"]="$low_priority_frequency"
  ["--low-priority-tokens-limit"]="$low_priority_tokens_limit"
  ["--ssl-keyfile"]="$ssl_keyfile"
  ["--ssl-certfile"]="$ssl_certifile"
  ["--ssl-ca-certs"]="$ssl_ca_certs"
  ["--ssl-cert-reqs"]="$ssl_cert_reqs"
  ["--dtype"]="$dtype"
  ["--api-key"]="$api_key"
  ["--guided-decoding-disable-any-whitespace"]="true"
  ["--guided-decoding-backend"]="xgrammar"
  ["--scheduling-policy"]="priority"
  ["--reasoning-parser"]="glm45"
  ["--tool-call-parser"]="glm47"
)



python_cmd="python3 -m vllm.entrypoints.openai.api_server --trust-remote-code --enable-expert-parallel --enable-auto-tool-choice"

for key in ${!params[@]}; do
  value=${params[$key]}
  if [ -n "$value" ]; then
    python_cmd+=" $key $value"
  fi
done
eval $python_cmd