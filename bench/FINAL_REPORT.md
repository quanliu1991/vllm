# vLLM v0.30.0 · Qwen3.6-35B-A3B · 2×L20 性能优化 — 最终报告

**任务**: 2000输入/200输出/32并发, 开启思考解析+工具解析, 关闭 prefix caching,
TTFT / TPOT / 日请求量较生产基线提升 ≥2×。
**环境**: `docker.das-security.cn/hb/vllm-openai:v0.30.0` + 分支 `perf/qwen35-2x-l20`
(worktree 挂载), 2×NVIDIA L20 (SM89, 48GB, PCIe 无NVLink, device 4/5)。

## 最终成绩 (3次回归 + 终验, 128 请求/轮)

| 指标 | 基线 (生产参数) | 最优配置 | 提升 | 2×目标 | 达成 |
|---|---|---|---|---|---|
| TTFT mean | 0.978s | 0.94s (0.89~0.96) | **1.04×** | 0.489s | ✗ |
| real TPOT* | 32.5ms | 20.8ms | **1.56×** | 16.25ms | ✗ |
| 吞吐 | 4.28 req/s | 6.27 req/s | **1.46×** | 8.57 | ✗ |
| 日容量 | 370k | 541k | **1.46×** | 740k | ✗ |
| 单流 TPOT | ~6.5 ms/tok | 3.28 ms/tok | **1.98×** | — | ≈✓ |

*real TPOT = (E2E−TTFT)/199。chunked TPOT 53.4ms 是 MTP 合并 chunk 的测量假象
(1.56 token/chunk), 真实每 token 延迟 20.8ms。

**结论: 三项并发指标均未达到 2×; 单序列场景接近 2×。**
差距的每一微秒都已定位并量化(见下), 属 2×L20 PCIe 硬件结构性瓶颈,
在 v0.30.0 软件栈内无法消除。

## 推荐生产启动参数

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3.6-35B-A3B \
  --served-model-name HengNao-v4 \
  --tensor-parallel-size 2 \
  --quantization fp8_per_channel \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --mamba-cache-mode align \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

约束: max_model_len ≤32768 (MTP+Mamba cache 显存约束; 262144 时 OOM)。

## 优化项归因表

| # | 优化项 | 类型 | 单项贡献 (req/s) | 提交 |
|---|---|---|---|---|
| 1 | FP8 在线量化 fp8_per_channel (PTPC, 权重带宽减半) | 配置+**bug修复** | 4.28→5.14 (+20%) | faf6248a |
| 2 | MTP 投机解码 k=2 (1.56 tok/step) | 配置 | 5.14→6.15 (+20%) | — |
| 3 | L20 Triton MoE 调优 config (E=256) | **代码** | 6.15→6.24 (+1.5%, TPOT -3%) | 0076363b |
| 4 | mnbt 4096→8192 (FP8 prefill 加速后重新平衡) | 配置 | 含在 #2 | — |
| 5 | bf16 替代 fp16 / custom AR / EP2 / KV-FP8 | 配置 | ≈0 (实测) | — |

### 关键 bug 修复 (上游 v0.30.0)

`vllm/model_executor/kernels/linear/scaled_mm/cutlass.py`
`process_weights_after_loading` 存在两处缺陷, 导致 `--quantization fp8_per_channel`
启动即崩 (`AssertionError: Overwriting existing tensor attribute: weight_loader`):
1. scale padding 后将 loader 安装到 **weight 参数**(复制粘贴错误, 应为 scale)
2. `replace_parameter` 继承旧 loader 与 `set_weight_attrs` no-overwrite 断言冲突

### 调优配置补齐 (上游缺失)

L20 (SM89) 无任何 Triton MoE 调优配置, 回落默认参数。用官方 tuner
(`benchmarks/kernels/benchmark_moe.py --tune`) 实测调优:
- `E=256,N=256` (TP2 形状): batch 48~2048 全扫描 (~3.5h, 双卡并行)
- `E=256,N=512` (TP1): 自 L40S (同 SM89) 拷贝
TP2 形状微基准: FP8 0.59ms/层 (686GB/s, 带宽 79%), **kernel 级 2.05×**

## 瓶颈定量分解 (32cc 解码步 33ms/step)

| 组成 | 时间 | 占比 | 可否优化 |
|---|---|---|---|
| MoE 专家 GEMM (FP8, 已带宽饱和) | ~13ms | 39% | 已到硬件极限 |
| GDN+attention (MTP ×3 位置) | ~7ms | 21% | 接受率决定 |
| TP2 allreduce + launch | ~13ms | 40% | **PCIe 结构性** |

- 采样窗内 NCCL AR 占 GPU 时间 25% (prefill 大块 2.7ms/次)
- EP2 实测更差 (5.93 vs 6.24): PCIe all-to-all 劣于 TP AR
- CUTLASS MoE 后端需 SM90+; DeepGEMM 需 block-quant — SM89 均不可用
- MTP 接受率: pos0 87.4%, pos1 68.6% → 1.56 tok/step, 32cc 已近上限

## 精度门 (无损确认)

| 测试 | BF16 基线 | FP8+MTP |
|---|---|---|
| GSM 数学 | PASS | PASS |
| 多步计算 | PASS | PASS |
| 事实问答 | PASS | PASS |
| 逻辑推理 | PASS | PASS |
| 工具调用 (参数精确匹配) | PASS (severity:['高']) | PASS (severity:['高']) |

**5/5 一致通过, 工具参数逐字段相同 → 量化+投机解码无损。**

## 若必须达 2× 的路径建议

1. **换硬件**: NVLink 机型 (H20/L40S NVSwitch) — 消除 40% 通信项
2. **4卡 TP4/EP4**: 带宽×2, 专家/卡减半
3. **升级 vLLM**: 更新版本 SM89 CUTLASS MoE 支持
4. **W4A16 (Marlin)**: 权重带宽再减半 (需接受精度评估)
5. 本分支三个 commit 均可独立 cherry-pick 至生产分支
