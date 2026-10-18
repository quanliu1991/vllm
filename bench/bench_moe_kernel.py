#!/usr/bin/env python3
"""Micro-benchmark: Triton fused MoE BF16 vs FP8 on SM89 (L20).

Simulates one decode step of Qwen3.6-35B-A3B: 32 seqs x topk=8 = 256
expert-token pairs over 256 experts, one layer. This is the memory-bound
path that dominates TPOT.
"""

import sys
import time

import torch

sys.path.insert(0, "/workspace/vllm")

from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts  # noqa: E402

E, TOPK, H, I = 256, 8, 2048, 512
NUM_TOKENS = 32


def make_inputs(dtype):
    x = torch.randn(NUM_TOKENS, H, dtype=torch.bfloat16, device="cuda")
    w1 = (
        torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device="cuda") * 0.02
    ).to(dtype)
    w2 = (torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda") * 0.02).to(dtype)
    # token i picks experts i*topk .. i*topk+7 -> every expert used exactly once
    topk_ids = (
        torch.arange(NUM_TOKENS * TOPK, device="cuda").reshape(NUM_TOKENS, TOPK) % E
    )
    topk_weights = torch.full((NUM_TOKENS, TOPK), 1.0 / TOPK, device="cuda")
    return x, w1, w2, topk_ids, topk_weights


def bench(label, dtype, quant_config, iters=100):
    x, w1, w2, topk_ids, topk_weights = make_inputs(dtype)

    def run():
        return fused_experts(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            quant_config=quant_config,
        )

    for _ in range(20):
        run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    print(f"{label}: {ms:.3f} ms/layer")
    return ms


def main():
    torch.cuda.set_device(0)
    bf16_ms = bench("BF16", torch.bfloat16, None)

    # FP8 weights, dynamic per-token activation quant (PTPC-style)
    x, w1, w2, topk_ids, topk_weights = make_inputs(torch.float8_e4m3fn)
    n_w13 = w1.shape[1]
    w1_scale = torch.ones(E, n_w13, 1, device="cuda", dtype=torch.float32)
    w2_scale = torch.ones(E, H, 1, device="cuda", dtype=torch.float32)
    qc = fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=None,
        a2_scale=None,
        per_act_token_quant=True,
        per_out_ch_quant=True,
    )
    fp8_ms = bench("FP8 PTPC", torch.float8_e4m3fn, qc)

    print(f"speedup: {bf16_ms / fp8_ms:.2f}x")
    bytes_bf16 = (E * (2 * I * H + H * I)) * 2 / 1e9
    bytes_fp8 = (E * (2 * I * H + H * I)) * 1 / 1e9
    print(f"expert bytes/layer: bf16={bytes_bf16:.2f}GB fp8={bytes_fp8:.2f}GB")
    print(f"achieved BW: bf16={bytes_bf16 / (bf16_ms / 1e3):.0f}GB/s fp8={bytes_fp8 / (fp8_ms / 1e3):.0f}GB/s")


if __name__ == "__main__":
    main()
