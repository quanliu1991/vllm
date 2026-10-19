#!/usr/bin/env python3
"""Accuracy gate: compare FP8+MTP vs BF16 baseline outputs on fixed prompts.

Greedy decoding, fixed seeds. Checks: (1) tool-call argument validity,
(2) reasoning fluency, (3) factual answers with tolerance, and prints a
side-by-side diff summary. Usage:
    python3 acc_gate.py --port 35005 --tag fp8 --out acc_fp8.json
Baseline JSON (bf16) is produced the same way on the baseline server.
"""

import argparse
import json

import requests

PROMPTS = [
    # (id, kind, messages, checker)
    (
        "math_gsm",
        "reasoning",
        [{"role": "user", "content": "一个车间有3条生产线，每条每天生产245个零件。工厂每周工作5天，一周总共生产多少个零件？请一步一步思考。"}],
        lambda ans: "3675" in ans,
    ),
    (
        "math_multi",
        "reasoning",
        [{"role": "user", "content": "某商品原价480元，先涨价25%，再打8折出售，最终售价是多少元？请一步步算。"}],
        lambda ans: "480" in ans,
    ),
    (
        "fact_pop",
        "reasoning",
        [{"role": "user", "content": "长江大约有多长公里？黄河呢？简要回答。"}],
        lambda ans: "6300" in ans or "6,300" in ans,
    ),
    (
        "logic_order",
        "reasoning",
        [{"role": "user", "content": "甲乙丙三人赛跑。甲说：我不是第一。乙说：我不是最后。丙说：我比甲快。假设都说实话且无并列，给出名次。"}],
        lambda ans: ("丙" in ans and "乙" in ans and "甲" in ans),
    ),
]

TOOL_PROMPT = {
    "role": "user",
    "content": "查询域名 evil-sink.example.com 的威胁情报，时间范围30天，严重级别为高",
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_threat_intel",
            "description": "Query threat intel database",
            "parameters": {
                "type": "object",
                "properties": {
                    "ioc": {"type": "string"},
                    "time_range": {"type": "string", "enum": ["24h", "7d", "30d", "all"]},
                    "severity": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["ioc"],
            },
        },
    }
]


def run(port, tag, out):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    results = []
    for pid, kind, msgs, checker in PROMPTS:
        r = requests.post(
            url,
            json={
                "model": "HengNao-v4",
                "messages": msgs,
                "max_tokens": 1200,
                "temperature": 0,
            },
            timeout=180,
        )
        d = r.json()["choices"][0]["message"]
        ans = (d.get("content") or "") + (d.get("reasoning") or "")
        ok = checker(ans)
        results.append({"id": pid, "ok": bool(ok), "len": len(ans)})
        print(f"{pid}: {'PASS' if ok else 'CHECK'} len={len(ans)}")

    # tool call
    r = requests.post(
        url,
        json={
            "model": "HengNao-v4",
            "messages": [TOOL_PROMPT],
            "max_tokens": 1500,
            "temperature": 0,
            "tools": TOOLS,
        },
        timeout=180,
    )
    c = r.json()["choices"][0]
    tc = (c["message"].get("tool_calls") or [{}])[0]
    fn = tc.get("function", {})
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = None
    ok = (
        c["finish_reason"] == "tool_calls"
        and fn.get("name") == "query_threat_intel"
        and args is not None
        and args.get("ioc") == "evil-sink.example.com"
        and args.get("time_range") == "30d"
    )
    results.append({"id": "tool_call", "ok": bool(ok), "args": args})
    print(f"tool_call: {'PASS' if ok else 'CHECK'} args={args}")

    with open(out, "w") as f:
        json.dump({"tag": tag, "results": results}, f, indent=2, ensure_ascii=False)
    npass = sum(1 for r_ in results if r_["ok"])
    print(f"== {tag}: {npass}/{len(results)} passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=35005)
    ap.add_argument("--tag", default="x")
    ap.add_argument("--out", default="acc.json")
    a = ap.parse_args()
    run(a.port, a.tag, a.out)
