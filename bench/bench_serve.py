#!/usr/bin/env python3
"""Benchmark harness: 2000-in/200-out, 32 concurrent, reasoning+tool parsing on.

Emulates the production workload (tool-calling assistant with thinking):
- System prompt with a realistic tool schema (~600 tokens)
- User question (~1400 tokens of context/document)
- 200 output tokens (0.2 temperature for near-deterministic reasoning traces)

Reports: mean/p50/p99 TTFT, mean/p99 TPOT (inter-token latency), throughput,
request rate, and derived daily request capacity.
"""

import argparse
import json
import math
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_threat_intel",
        "description": (
            "Query the threat intelligence database for indicators of "
            "compromise, malware families, and threat actors. Supports "
            "filtering by time range, severity, and attack vector. Returns "
            "structured results with confidence scores and source citations "
            "that should be summarized for the analyst."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ioc": {"type": "string", "description": "IP, domain, hash or URL"},
                "time_range": {"type": "string", "enum": ["24h", "7d", "30d", "all"]},
                "severity": {"type": "array", "items": {"type": "string"}},
                "include_mitre": {"type": "boolean"},
            },
            "required": ["ioc"],
        },
    },
}

FILLER = (
    "在进行安全事件调查时，分析师需要综合多方数据源进行研判。"
    "以下是一段与当前事件相关的历史处置记录与背景资料，请仔细阅读后回答问题。"
) * 24


def build_prompt():
    system = {
        "role": "system",
        "content": (
            "You are HengNao, an enterprise security assistant. "
            "Think step by step inside <think>...</think>, then answer concisely. "
            "Use the provided tools when the question requires live data."
        ),
    }
    user = {
        "role": "user",
        "content": (
            FILLER
            + "\n\nGiven the above incident context, identify the most likely "
            "attack vector and produce a remediation checklist. If live threat "
            "intel would change the conclusion, call the tool."
        ),
    }
    return [system, user]


def one_request(endpoint, model, max_tokens, timeout, seed):
    rid = str(uuid.uuid4())
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    timestamps = []
    finish = None
    try:
        with requests.post(
            endpoint,
            json={
                "model": model,
                "messages": build_prompt(),
                "tools": [TOOL_SCHEMA],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "stream": True,
                "seed": seed,
            },
            stream=True,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data:"):
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"]
                    except Exception:
                        continue
                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning") or ""
                    if content or reasoning:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        timestamps.append(now)
                        chunks += 1
                elif line.startswith(b"event:") or line.startswith(b":"):
                    continue
        finish = time.perf_counter() - t0
    except Exception as e:
        return {"error": str(e), "ttft": None, "tpots": [], "total": None}
    tpots = [
        (timestamps[i + 1] - timestamps[i]) for i in range(len(timestamps) - 1)
    ]
    return {"ttft": ttft, "tpots": tpots, "total": finish, "chunks": chunks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=35005)
    ap.add_argument("--model", default="HengNao-v4")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    endpoint = f"http://{args.host}:{args.port}/v1/chat/completions"
    # warmup
    one_request(endpoint, args.model, 8, args.timeout, seed=0)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(
            ex.map(
                lambda i: one_request(
                    endpoint, args.model, args.max_tokens, args.timeout, seed=i
                ),
                range(args.num_prompts),
            )
        )
    wall = time.perf_counter() - t0

    errs = [r for r in results if r.get("error")]
    ok = [r for r in results if not r.get("error") and r["ttft"] is not None]
    ttfts = sorted(r["ttft"] for r in ok)
    all_tpots = [t for r in ok for t in r["tpots"]]
    totals = [r["total"] for r in ok]

    def pct(xs, p):
        if not xs:
            return float("nan")
        i = min(len(xs) - 1, max(0, math.ceil(p * len(xs)) - 1))
        return xs[i]

    out = {
        "tag": args.tag,
        "concurrency": args.concurrency,
        "num_prompts": args.num_prompts,
        "completed": len(ok),
        "errors": len(errs),
        "ttft_mean_s": statistics.mean(ttfts) if ttfts else None,
        "ttft_p50_s": pct(ttfts, 0.5),
        "ttft_p99_s": pct(ttfts, 0.99),
        "tpot_mean_ms": statistics.mean(all_tpots) * 1000 if all_tpots else None,
        "tpot_p99_ms": pct(sorted(all_tpots), 0.99) * 1000 if all_tpots else None,
        "e2e_mean_s": statistics.mean(totals) if totals else None,
        "req_throughput_rps": len(ok) / wall,
        "output_tok_per_s": sum(r["chunks"] for r in ok) / wall,
        "wall_s": wall,
        "daily_capacity_req": len(ok) / wall * 86400,
    }
    print(json.dumps(out, indent=2))
    with open(f"/workspace/vllm/bench/results_{args.tag}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
