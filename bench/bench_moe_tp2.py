#!/usr/bin/env python3
"""TP2-shape MoE kernel microbench: per-card E=256 experts, I/tp=256.

Mimics one decode step under 32-seq concurrency with MTP k=2:
M = 32 seqs x 3 positions = 96 tokens x topk 8 = 768 expert-token pairs
spread over 256 experts (~3 tokens/expert). Compares BF16 vs FP8 PTPC
Triton fused MoE with the tuned L20 configs.
"""

import sys
import time

import torch

sys.path.insert(0, "/workspace/vllm")

from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts  # noqa: E402

# TP2 per-card shapes
E, TOPK, H, I_TP = 256, 8, 2048, 256


def make_inputs(dtype, num_tokens):
    x = torch.randn(num_tokens, H, dtype=torch.bfloat16, device="cuda")
    w1 = (
        torch.randn(E, 2 * I_TP, H, dtype=torch.bfloat16, device="cuda") * 0.02
    ).to(dtype)
    w2 = (
        torch.randn(E, H, I_TP, dtype=torch.bfloat16, device="cuda") * 0.02
    ).to(dtype)
    # uniform routing: every expert gets num_tokens*topk/E pairs
    ids = torch.randperm(num_tokens * TOPK, device="cuda") % E
    topk_ids = ids.reshape(num_tokens, TOPK)
    topk_weights = torch.full((num_tokens, TOPK), 1.0 / TOPK, device="cuda")
    return x, w1, w2, topk_ids, topk_weights


def bench(label, dtype, quant_config, num_tokens, iters=100):
    x, w1, w2, topk_ids, topk_weights = make_inputs(dtype, num_tokens)

    def run():
        return fused_experts(
            x, w1, w2, topk_weights, topk_ids, quant_config=quant_config
        )

    for _ in range(20):
        run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    print(f"{label} M={num_tokens}: {ms:.3f} ms/layer")
    return ms


def main():
    torch.cuda.set_device(0)
    for m in (32, 64, 96, 128):
        bf16 = bench("BF16 ", torch.bfloat16, None, m)
        # FP8 PTPC
        x, w1, w2, topk_ids, topk_weights = make_inputs(
            torch.float8_e4m3fn, m
        )
        w1_scale = torch.ones(E, 2 * I_TP, 1, device="cuda", dtype=torch.float32)
        w2_scale = torch.ones(E, H, 1, device="cuda", dtype=torch.float32)
        qc = fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=None,
            a2_scale=None,
            per_act_token_quant=True,
            per_out_ch_quant=True,
        )
        fp8 = bench("FP8  ", torch.float8_e4m3fn, qc, m)
        bytes_bf16 = E * (2 * I_TP * H + H * I_TP) * 2 / 1e9
        bytes_fp8 = E * (2 * I_TP * H + H * I_TP) * 1 / 1e9
        print(f"  speedup {bf16/fp8:.2f}x  BW bf16={bytes_bf16/(bf16/1e3):.0f} fp8={bytes_fp8/(fp8/1e3):.0f} GB/s\n")


if __name__ == "__main__":
    main()
