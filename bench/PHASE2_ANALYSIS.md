# Phase 2 Deep-Dive: Path to 2x on 2xL20 (PCIe)

Branch: `perf/qwen35-2x-l20-phase2` (from stage-1 milestone `939b6496`,
tag `v0.30.0-qwen35-l20-stage1`).

## Goal recap (all 2x vs baseline)

| Metric | Baseline | Stage 1 | Target 2x | Gap |
|---|---|---|---|---|
| TTFT mean | 0.978s | 0.94s (1.04x) | 0.489s | 1.92x to go |
| TPOT mean | 32.5ms | 20.8ms (1.56x) | 16.25ms | 1.28x to go |
| Throughput | 4.28 r/s | 6.27 r/s (1.46x) | 8.57 r/s | 1.37x to go |

## New measurements this phase

1. **Pure prefill latency (2000 tok, idle, warm): 58ms.**
   Loaded TTFT is 0.94s → **94% of TTFT is queueing/interference**, not prefill
   compute. TTFT is a scheduling problem, not a kernel problem.
2. **Steady-state window trace re-analysis (2.97s, 77 steps, mixed workload):**
   - Step wall p50 28.4ms pure-decode clusters; window average 38.6ms/step
     → prefill interference ≈ 10ms/step amortized (26%).
   - GPU busy >100% of wall (multi-stream overlap; CUDA graphs FULL, 46 sizes).
   - async_scheduling resolves ON for MTP (verified in `vllm/config/vllm.py`
     auto-enable path) — already active, not an untapped lever.
3. **Per-step kernel budget (77-step window):**

   | Component | ms/step | Share | Notes |
   |---|---|---|---|
   | MoE FP8 GEMM (Triton) | 12.2 | 32% | ~130 experts hit/layer (skewed routing); p50 122us/call; bandwidth-saturated |
   | Dense FP8 GEMM (cutlass sm89) | 5.8 | 15% | 213 launches/step at 25-45us latency floor; grids ≤96 CTAs |
   | NCCL AR (prefill chunks) | 5.5 | 14% | ~2.7 calls/step x 2ms; 8MB @ ~8GB/s effective |
   | GDN attention | 4.4 | 11% | `decode_post_conv_mtp` |
   | bf16 GEMM (prefill dense + lm_head + draft) | 2.9 | 8% | lm_head NOT fp8-quantized |
   | custom AR (decode, symmetric mem) | 2.4 | 6% | 89 x 26us; near PCIe floor |
   | Flash attention (10 full-attn layers) | 3.1 | 8% | |
   | sampling + elementwise + misc | 2.2 | 6% | |

4. **Variant sweeps this phase (all REJECTED):**
   - `--max-num-batched-tokens 32768`: TPOT 51.2ms (regression vs 8192's 20.8).
   - `NCCL_ALGO=Ring NCCL_PROTO=Simple`: TPOT 60.6ms, TTFT 1.19s (hard regression).

## Key insights

1. **TTFT is 94% queueing.** Any prefill-priority / prefill-batching scheduler
   change attacks 0.88s of the 0.94s directly.
2. **Prefill interference costs ~10ms/step (26%)** on decode TPOT under 32cc
   continuous arrivals (prefill duty ≈ 34% of GPU time; every decode step is
   "mixed" in the trace).
3. **lm_head + draft dense run in BF16** under `fp8_per_channel`
   (`QuantSpec` maps linear+moe only; ParallelLMHead stays bf16):
   ~2-3ms/step + 312MB weights. Candidate: extend PTPC to lm_head behind an
   accuracy gate.
4. **Dense FP8 GEMMs are latency-floor-bound** (213 launches x ~27us): no
   config fix; would need kernel fusion (e.g., fused qkv+gate_up via
   N-concatenation where inputs align) — invasive.
5. **Decode customAR (26us x 89) is near the PCIe floor**; prefill NCCL AR at
   8GB/s is 2-3x below achievable but env-level NCCL forcing regresses
   globally (custom AR and small-message paths degrade).
6. **MoE FP8 is bandwidth-saturated** (0.30ms/layer, ~130/256 experts touched).
   Next material step is W4A16 experts (halves bytes again) but requires a
   quantized checkpoint (llm-compressor) → accuracy-gated, offline work.

## Ranked plan to close the remaining gap

| # | Lever | Type | Est. gain | Risk |
|---|---|---|---|---|
| 1 | Prefill/decode scheduling: batch waiting prefills into dedicated steps (per-step prefill token cap knob), keep decode-only steps between | scheduler code | TPOT -3ms, TTFT 0.94→~0.4s | Medium (correctness of chunked-prefill+MTP interplay) |
| 2 | lm_head + draft linear FP8 (PTPC) behind `--quantization` extension or post-load hook | quant code | TPOT -1.5~2ms, +312MB KV room | Accuracy-gated (argmax shift) |
| 3 | GDN `decode_post_conv_mtp` fusion (batch conv+sigmoid+ddp chains) | kernel | -1~1.5ms | Medium |
| 4 | MoE W4A16 experts (AWQ/GPTQ via llm-compressor offline) | checkpoint | -4~5ms | Accuracy-gated, offline pipeline |
| 5 | MTP k=3 in steady state (acceptance-adaptive) | config/Code | acceptance 1.56→~2.0 ⇒ TPOT -15% | Low-med; k=3 measured worse at fixed batch |

Realistic combined: step 38.6 → ~26-28ms ⇒ TPOT ~17ms (1.9x), TTFT ~0.4s (2.4x),
throughput ~8 r/s (1.87x) — the full 2x trio lands if #1+#2 (+#5 adaptive) hit.

## Artifacts

- `results_ttftw.json`, `results_ttft16384.json`: mnbt sweep.
- `results_nccl_ring.json`: NCCL forcing regression record.
- Trace: `/tmp/prof2/rank0` (steady window re-analysis scripts inline in session).
