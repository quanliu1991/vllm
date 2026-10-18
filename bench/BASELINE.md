# 基线测试记录（阶段 0）

日期: 2026-10-18
环境: perf030 容器 (docker.das-security.cn/hb/vllm-openai:v0.30.0 + worktree 代码挂载)
硬件: 2× NVIDIA L20 (device 4,5), --shm-size 32g
模型: /opt/models/Qwen3.6-35B-A3B (Qwen3_5MoeForConditionalGeneration, BF16 权重)

## 服务参数（生产等价, 来自 deploy_chat_3_5.sh + hb-serve run.sh）

- TP=2, dtype=float16, gpu-mem-util=0.9, max-model-len=262144
- max-num-batched-tokens=4096 (deploy 脚本覆盖值; run.sh 默认 64000)
- max-num-seqs: vllm 默认 (256)
- VLLM_DISABLE_CUSTOM_ALL_REDUCE=true (生产 env)
- VLLM_USE_FLASHINFER_SAMPLER=0
- --no-enable-prefix-caching (用户要求, 所有测试统一)
- --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice
  (生产 run.sh reasoning_parser_map: HengNao-3_5-35ba3b-chat -> qwen3.5→qwen3,
   tool_call_parser 默认 qwen3_coder)

## 负载

- 2000 input tokens (system+tool schema+filler ~1400 + question), 200 output
- 32 并发, 128 请求, temperature=0.2, stream
- 基准脚本: bench/bench_serve.py (自研, 记录 TTFT/TPOT/E2E/吞吐)

## 结果

| 指标 | run1 | run2 (稳态) | 采用值 |
|---|---|---|---|
| TTFT mean (s) | 2.595 | 0.978 | 0.978 (run2; run1 含冷启动首个batch的chunked prefill排队) |
| TTFT p50 (s) | 0.822 | 0.808 | 0.808 |
| TTFT p99 (s) | 15.48 | 2.848 | 2.848 |
| TPOT mean (ms) | 40.4 | 32.5 | 32.5 |
| TPOT p99 (ms) | 287.0 | 286.7 | 286.7 |
| E2E mean (s) | 10.64 | 7.45 | 7.45 |
| 请求吞吐 (req/s) | 3.00 | 4.28 | 4.28 |
| 输出吞吐 (tok/s) | 600 | 857 | 857 |
| 日处理请求数 (推算) | 259k | 370k | 370k |

注意:
- run1 的 p99 TTFT 15.5s 是首批请求叠加 chunked prefill 排队所致 (128×2000tok / 4096 token/轮)
- TPOT p99 ~287ms 异常高, 与 32 并发下 prefill chunk 与 decode 交替有关 (chunked prefill 默认开启时每个 prefill chunk 间隙 decode 停滞)
- run2 为稳态值, 后续优化对比以 run2 口径为准（同样先跑一轮预热再采数）

## 2x 目标（基于 run2）

- TTFT mean ≤ 0.489 s
- TPOT mean ≤ 16.25 ms
- 日请求 ≥ 740k (吞吐 ≥ 8.57 req/s)
