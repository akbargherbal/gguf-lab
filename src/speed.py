"""Throughput measurement on top of llama.cpp's own timings.

llama-server reports `timings.prompt_ms` (prefill) and
`timings.predicted_per_second` (decode) for each request. Those are the numbers
that matter, because they separate "slow to read the prompt" from "slow to
write" — two different failure modes on a T4.

When a build omits `timings`, the same fields are derived from wall-clock time
and `usage.completion_tokens`; the report marks which source was used so a
comparison never mixes the two silently.
"""

from __future__ import annotations

import time

PROMPT = (
    "Explain in two short paragraphs why a small quantized language model can "
    "still be useful for editing text on a laptop, and name one thing it is "
    "likely to get wrong."
)


def one_run(server, prompt: str = PROMPT, max_tokens: int = 256) -> dict:
    """One timed completion, normalised to seconds and tokens/second."""
    started = time.time()
    reply = server.chat(
        [{"role": "user", "content": prompt}], max_tokens=max_tokens
    )
    wall = time.time() - started
    timings = reply.get("timings") or {}
    usage = reply.get("usage") or {}

    completion = usage.get("completion_tokens") or timings.get("predicted_n")
    prompt_tokens = usage.get("prompt_tokens") or timings.get("prompt_n")
    source = "timings" if timings else "wallclock"

    prefill_s = None
    if timings.get("prompt_ms") is not None:
        prefill_s = round(float(timings["prompt_ms"]) / 1000.0, 3)
    decode_tok_s = None
    if timings.get("predicted_per_second"):
        decode_tok_s = round(float(timings["predicted_per_second"]), 1)
    elif completion and wall > 0:
        decode_time = max(1e-6, wall - (prefill_s or 0.0))
        decode_tok_s = round(float(completion) / decode_time, 1)

    e2e_tok_s = None
    if completion and wall > 0:
        e2e_tok_s = round(float(completion) / wall, 1)

    return {
        "wall_s": round(wall, 2),
        "prefill_s": prefill_s,
        "decode_tok_s": decode_tok_s,
        "e2e_tok_s": e2e_tok_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "output_words": len(reply["text"].split()),
        "source": source,
    }


def summarize(runs: list[dict]) -> dict:
    """Aggregate repeated runs; means plus the slowest/fastest decode rate."""

    def mean(key: str) -> float | None:
        values = [r[key] for r in runs if r.get(key) is not None]
        return round(sum(values) / len(values), 2) if values else None

    decodes = [r["decode_tok_s"] for r in runs if r.get("decode_tok_s")]
    return {
        "runs": len(runs),
        "mean_wall_s": mean("wall_s"),
        "mean_prefill_s": mean("prefill_s"),
        "mean_decode_tok_s": mean("decode_tok_s"),
        "mean_e2e_tok_s": mean("e2e_tok_s"),
        "min_decode_tok_s": round(min(decodes), 1) if decodes else None,
        "max_decode_tok_s": round(max(decodes), 1) if decodes else None,
        "source": runs[0]["source"] if runs else None,
        "completion_tokens": runs[0]["completion_tokens"] if runs else None,
    }


def run_speed(server, runs: int = 3, max_tokens: int = 256, prompt: str = PROMPT,
              sampler=None) -> dict:
    """Warm up once (model load, CUDA graphs), then time `runs` completions."""
    server.start()
    measurements: list[dict] = []
    for _ in range(max(1, runs)):
        measurements.append(one_run(server, prompt=prompt, max_tokens=max_tokens))
    report = summarize(measurements)
    if sampler is not None:
        report["peak_ram_gib"] = round(sampler.peak_ram_gib, 1)
        report["peak_vram_gib"] = round(sampler.peak_vram_gib, 1)
    report["raw"] = measurements
    return report
