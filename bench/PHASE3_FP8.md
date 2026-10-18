# 阶段3 精度无损量化 — 中期记录

## 权重 FP8 在线量化 (fp8_per_channel / PTPC)

### 上游 Bug 修复 (commit faf6248a)

`--quantization fp8_per_channel` 在 v0.30.0 启动即崩:

```
AssertionError: Overwriting existing tensor attribute: weight_loader
```

根因 (`vllm/model_executor/kernels/linear/scaled_mm/cutlass.py`):
1. **复制粘贴错误**: `process_weights_after_loading` 对 scale padding 后,
   `set_weight_attrs(getattr(layer, weight_name), ...)` 应为 `weight_scale_name`
   — 对同一 weight 参数二次安装 loader, 命中 no-overwrite 断言
2. **replace_parameter 携带旧 loader**: 被替换参数自动继承旧 weight_loader,
   与后续 set_weight_attrs 冲突

修复: 抽出 `_install_padded_weight_loader()` — 先 `del param.weight_loader`
再 set; 并修正目标参数名。

### 正确性验证 (FP8 PTPC)

- 推理输出: 正常中文推理流 (`reasoning` 字段完整连贯)
- 工具调用: `finish_reason=tool_calls`, 参数 JSON 完全正确
  (`{"ioc": "185.220.101.5", "time_range": "7d"}`)

### 性能 (32cc/2000in/200out, --no-enable-prefix-caching)

| 变体 | TTFT mean | real TPOT* | req/s | 日容量 |
|---|---|---|---|---|
| baseline (fp16, mnbt4096) | 0.978 | 32.5ms | 4.28 | 370k |
| kv-fp8 only | 1.014 | 33.4ms | 4.19 | 362k |
| **wfp8 (fp8_per_channel)** | 0.739 | 27.4ms | 5.14 | 444k |
| wfp8+mtp k=2 (mnbt8192) | 0.946 | 21.2ms | **6.15** | 532k |
| wfp8+mtp k=3 (mnbt8192) | 0.899 | 22.0ms | 6.02 | 520k |
| wfp8+mtp k=1 (mnbt16384) | 1.237 | 20.9ms | 5.90 | 510k |

*real TPOT = (E2E−TTFT)/199; chunked TPOT 在 MTP 下失真(多 token 合并 chunk)

- 单流 greedy: 3.28 ms/tok (baseline ~6.5) — 单序列视角 2× 达成
- KV-FP8 单独开启无收益 (本场景 KV 不是瓶颈; 显存富余)
- MTP k=2 + mnbt8192 为当前最优组合

## 关键发现: Triton MoE tuning config 缺失

microbenchmark (单卡 L20, E=256 全专家, 32 tok × topk8):

```
BF16:               2.228 ms/layer (默认 config, 未调优)
FP8 PTPC (默认):    1.179 ms/layer  → 1.89x
FP8 PTPC (L40S tune): 1.131 ms/layer → 1.97x, 712 GB/s (≈L20 带宽 82%)
```

- L20 与 L40S 同为 SM89, 上游已有 `E=256,N=512,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json`
- **TP2 部署实际查找键 = E=256, N=2048 (w2.shape[1])**, 无任何 L20/L40S config → 回落默认配置
- 已补 `E=256,N=512,device_name=NVIDIA_L20,dtype=fp8_w8a8.json` (L40S 拷贝)
- 正在跑 `benchmarks/kernels/benchmark_moe.py --tune --batch-size 48 64 96 128 256 512 2048`
  生成 TP2 形状的 L20 调优配置 (E=256,N=2048)

## SM89 后端选型结论

- CUTLASS MoE backend: **SM90+ only**, `--moe-backend cutlass` 明确拒绝 SM89
- DeepGEMM: 存在但 `_valid_deep_gemm` 对 N≤512 形状禁用; 且 PTPC 非 block-quant
- 结论: SM89 上 FP8 MoE 只能走 Triton grouped GEMM → 调优 config 是必经之路

## 待完成

- [ ] TP2 shape (E=256,N=2048) tuning configs 生成并验证端到端收益
- [ ] 精度门: gsm8k / 内部评测 vs BF16 baseline
- [ ] 与 MTP 叠加回归
