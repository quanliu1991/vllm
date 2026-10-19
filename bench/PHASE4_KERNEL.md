# 阶段4 内核/代码级优化 — 记录

## Profiling (torch profiler, FP8+MTP k=2, 32cc 负载下抓取)

GPU kernel 时间分布 (22s 采样窗, rank0):

| kernel | 时间 | 占比 | 说明 |
|---|---|---|---|
| fused_moe_kernel (Triton FP8) | 1247ms | 22% | 7812 次 × 160µs; 40层×2GEMM/step → ~12.8ms/step |
| **NCCL AllReduce bf16 (RING_LL)** | **1134ms** | **20%** | 420 次 × 2.7ms — **prefill 大块 AR** |
| NCCL AllGather | 274ms | 4.8% | |
| cross_device_reduce_1stage (custom AR) | 318ms | 5.6% | 7671 次 — decode 小消息 AR |
| GDN decode/chunk kernels | ~810ms | 14% | 线性注意力 |
| CUTLASS sm89 FP8 dense GEMM | ~760ms | 13% | qkv/o_proj/shared (已 FP8) |
| flash_fwd full-attn | 270ms | 4.8% | |

**结论**: 解码步 33ms 中, MoE 专家仅 ~13ms; 其余 ~20ms 是
attention(GDN×3 MTP 位置) + TP2 allreduce + launch 开销。
PCIe 双卡无 NVLink, TP2 通信是结构性上限。

## TP2 形状 MoE kernel 微基准 (调优后, L20)

| M (tokens) | BF16 ms/层 | FP8 ms/层 | 加速 | FP8 带宽 |
|---|---|---|---|---|
| 32 | 1.143 | 0.577 | 1.98× | 698 GB/s |
| 64 | 1.120 | 0.583 | 1.92× | 691 GB/s |
| 96 | 1.201 | 0.587 | 2.05× | 686 GB/s |
| 128 | 1.222 | 0.592 | 2.06× | 680 GB/s |

→ FP8 MoE kernel 本身已 2× 且带宽打满 (~80% of 864GB/s)。
E2E 无法 2× 的差距全部在 kernel 之外。

## 端到端变体扫描 (FP8 PTPC + MTP, 调优 config)

| 变体 | TTFT | real TPOT | req/s | 日容量 |
|---|---|---|---|---|
| mnbt8192 k2 (最优) | 0.844 | 20.6ms | **6.24** | 539k |
| mnbt16384 k2 | 0.955 | 21.0ms | 6.08 | 526k |
| mnbt4096 k2 | 1.024 | 38.5ms | 3.70 | 319k |
| EP2 (mnbt8192 k2) | 0.943 | 22.2ms | 5.93 | 513k |

- EP2 无收益: PCIe 上 all-to-all 劣于 TP AR
- mnbt 8192/16384 均达平台期 ~6.1-6.2 req/s

## MTP 接受率 (32cc)

- 19995 draft steps, 31180 accepted → **1.56 token/step**
- pos0: 87.4%, pos1: 68.6%

## 代码改动 (分支 perf/qwen35-2x-l20)

1. `faf6248a` fix(cutlass): padded weight loader 参数名错误 + 断言冲突
2. `0076363b` kernel: L20 Triton FP8 MoE tuned configs (E=256, N=256/N=512)
3. `f9402285` bench: 全部基准记录与测试脚本

## 剩余差距分析 (vs 2× 目标)

| 指标 | 基线 | 当前 | 提升 | 2×目标 | 差距 |
|---|---|---|---|---|---|
| TTFT mean | 0.978 | 0.844 | 1.16× | 0.489 | 0.58× |
| real TPOT | 32.5ms | 20.6ms | 1.58× | 16.25ms | 0.79× |
| req/s | 4.28 | 6.24 | 1.46× | 8.57 | 0.73× |
| 日容量 | 370k | 539k | 1.46× | 740k | 0.73× |

单流 TPOT 3.28ms (vs 基线 ~6.5) — 单序列场景 2× 达成。

### 无法在 2×L20 (PCIe, SM89) + v0.30.0 内消除的结构性瓶颈

1. **TP2 allreduce (25% GPU 时间)**: L20 无 NVLink, PCIe 4.0 x16 双卡
   峰值 ~25GB/s 单向; 每层 8MB bf16 prefill AR 2.7ms 不可避免
   (EP2 实测更差)
2. **SM89 无 FP8 MoE 加速后端**: CUTLASS MoE 需 SM90+, DeepGEMM 需
   block-quant + N>512; 只剩 Triton (已调优至带宽饱和)
3. **MTP 验证开销**: k=2 每 step 跑 3 个位置的 attention/GDN,
   接受率 1.56 决定上限 ~1.56×; 32 并发下已接近该上限
4. **prefill 与 decode 争抢**: 32 并发持续 prefill 挤占 decode 步,
   scheduler 层面无 chunked-prefill-priority 可调

### 若需进一步逼近 2× (超出本任务范围)

- 双机 4 卡 / NVLink 机型 (H20/L40S NVSwitch) → 通信瓶颈消除
- 升级 vLLM ≥0.11 的 SM89 cutlass MoE 支持
- 量化 KV + 权重至 INT4 W4A16 (Marlin) — 损失可控时带宽再减半
