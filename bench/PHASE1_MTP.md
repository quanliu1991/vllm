# 阶段1(配置) + 阶段2(MTP) 中期记录

## 配置扫描 (32cc/128req/2000in/200out, bf16, 温度0.2)

| 变体 | mnbt | TTFT mean | real TPOT* | req/s | 日容量 |
|---|---|---|---|---|---|
| baseline (fp16,no-AR) | 4096 | 0.978 | 32.5ms | 4.28 | 370k |
| cfg1 (bf16+AR) | 16384 | 1.760 | 27.5ms | 4.42 | 382k |
| cfg2 (bf16+AR) | 8192 | 1.350 | 30.2ms | 4.33 | 374k |
| cfg3 (bf16+AR) | 4096 | 0.990 | 32.5ms | 4.27 | 369k |
| *real TPOT = (E2E-TTFT)/199，chunk TPOT 在 MTP 下失真（多token合并chunk）*

- bf16+custom-allreduce 对本场景影响 ~0（L20 PCIe 双卡 TP2，消息小）
- mnbt 是 TTFT↔TPOT 权衡旋钮：16384 换 TPOT -15% 但 TTFT +80%
- **解码瓶颈=专家权重带宽**: 32并发×topk8→每层几乎激活全部256专家→每步~24GB/卡
  → 864GB/s HBM 下限 ~28ms，与实测吻合。单纯调参无法突破，需 FP8(带宽减半)+MTP(每步多token)

## MTP (qwen3_5_mtp, max_model_len=32768, max_num_seqs=128, mamba-cache-mode=align)

| 变体 | k | TTFT | real TPOT | req/s | 日容量 |
|---|---|---|---|---|---|
| mtp1 | 2 | 2.000 | 24.0ms | 4.45 | 384k |
| mtpk1 | 1 | 1.011 | 27.9ms | 4.83 | 417k |

- MTP 与 qwen3 思考解析 + qwen3_coder 工具解析完全兼容（finish_reason=tool_calls 正常）
- 单流 greedy: baseline 6.5ms/tok vs MTP k=2 10.9ms/tok（合并 chunk 前的每步开销上升，
  但每步提交~2.3 token → 有效 4.7ms/tok）
- k=1 吞吐最优(4.83)但 TPOT 不如 k=2；k=2 TPOT 最优
- 32并发下 MTP 对 TTFT 有小幅负影响（draft prefill 附加），mnbt 4096 时 k=1 TTFT 仍≈基线
- 注意: MTP 下 KV/Mamba cache 略缩（draft 占显存），max_model_len 需 ≤32768（262144 时 OOM）

## 结论与下一步

- 配置+MTP 叠加后: TPOT 24.0ms(-26%), 吞吐 4.83(+13%) —— 距 2× 目标(TPOT≤16.25ms, ≥8.57req/s)差距大
- 核心杠杆=FP8: 专家权重读取量减半 → 解码步时间近似减半; prefill 计算也提速
- 下一步: 阶段3 --kv-cache-dtype fp8 先行，再评估权重 FP8 (llm-compressor 离线量化)
