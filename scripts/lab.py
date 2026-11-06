#!/usr/bin/env python3
"""gguf-lab: screen small (often unofficial) GGUF models on a T4.

    python scripts/lab.py doctor
    python scripts/lab.py models
    python scripts/lab.py pull                      # default model
    python scripts/lab.py run --prompt "Hello"
    python scripts/lab.py bench speed
    python scripts/lab.py bench checks
    python scripts/lab.py bench checks --only json_object,arithmetic

Weights come from Hugging Face (or a GCS mirror). The llama.cpp binaries come
from the shared, arch-tagged cache staged by `plumbing/fetch_llamacpp.sh`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.checks import CHECKS, run_checks, score  # noqa: E402
from src.download import ensure_model  # noqa: E402
from src.metrics import MemorySampler  # noqa: E402
from src.registry import default_model, load_registry, resolve  # noqa: E402
from src.server import LlamaServer, find_server_bin  # noqa: E402
from src.speed import run_speed  # noqa: E402

DEFAULT_CACHE_GS = os.environ.get("GGUF_CACHE_GS", "")


def _registry(args):
    return load_registry(args.registry)


def _open_server(entry: dict, args) -> LlamaServer:
    model = ensure_model(
        entry, args.models_dir, cache_gs=args.cache_gs, log=print
    )
    return LlamaServer(
        model_path=model,
        server_bin=find_server_bin(),
        ctx_size=args.ctx,
        n_gpu_layers=args.ngl,
    )


def _save(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {path}", file=sys.stderr)


def cmd_models(args) -> int:
    registry = _registry(args)
    default = registry.get("default")
    rows = registry["models"]
    if args.json:
        print(json.dumps(registry, indent=2))
        return 0
    print(f"{'id':<30} {'quant':<8} {'GB':>5} {'VRAM~':>6}  {'official':<8} licence")
    print("-" * 92)
    for entry in rows:
        mark = "  (default)" if entry["id"] == default else ""
        official = "yes" if entry.get("official_conversion") else "no"
        print(
            f"{entry['id']:<30} {entry['quant']:<8} {entry['size_gb']:>5} "
            f"{entry.get('vram_gb', 0):>6}  {official:<8} "
            f"{entry['license']}{mark}"
        )
    print(f"\n{len(rows)} model(s). Weights land in {args.models_dir}.")
    return 0


def cmd_pull(args) -> int:
    registry = _registry(args)
    entries = registry["models"] if args.all else [resolve(registry, args.model)]
    for entry in entries:
        path = ensure_model(
            entry, args.models_dir, cache_gs=args.cache_gs, log=print
        )
        size = path.stat().st_size / 1024**3
        print(f"  {entry['id']}: {path} ({size:.2f} GiB)")
    return 0


def cmd_run(args) -> int:
    registry = _registry(args)
    entry = resolve(registry, args.model)
    server = _open_server(entry, args)
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    started = time.time()
    try:
        reply = server.chat(
            messages, max_tokens=args.max_tokens, temperature=args.temperature
        )
    finally:
        server.stop()
    elapsed = time.time() - started
    print(reply["text"])
    usage = reply.get("usage") or {}
    timings = reply.get("timings") or {}
    print(
        f"\n[{entry['id']}] {elapsed:.1f}s | "
        f"tokens {usage.get('completion_tokens', '?')} | "
        f"decode {timings.get('predicted_per_second', '?')} tok/s",
        file=sys.stderr,
    )
    return 0


def cmd_speed(args) -> int:
    registry = _registry(args)
    entry = resolve(registry, args.model)
    server = _open_server(entry, args)
    print(f"Benchmarking {entry['id']} ({entry['quant']}) ...", flush=True)
    sampler = MemorySampler()
    try:
        with sampler:
            report = run_speed(
                server, runs=args.runs, max_tokens=args.max_tokens, sampler=sampler
            )
    finally:
        server.stop()
    print(
        f"  wall {report['mean_wall_s']}s | prefill {report['mean_prefill_s']}s | "
        f"decode {report['mean_decode_tok_s']} tok/s "
        f"({report['min_decode_tok_s']}–{report['max_decode_tok_s']})"
    )
    print(
        f"  peak RAM {report.get('peak_ram_gib')} GiB | "
        f"peak VRAM {report.get('peak_vram_gib')} GiB | "
        f"source {report['source']}"
    )
    _save(args.out, {
        "model": {k: entry[k] for k in ("id", "file", "quant", "repo", "license")},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "speed": {k: v for k, v in report.items() if k != "raw"},
        "runs": report["raw"],
    })
    return 0


def cmd_checks(args) -> int:
    registry = _registry(args)
    entry = resolve(registry, args.model)
    only = [name.strip() for name in args.only.split(",")] if args.only else None
    server = _open_server(entry, args)
    sampler = MemorySampler()
    try:
        with sampler:
            results = run_checks(server, only=only, system=args.system)
    finally:
        server.stop()
    print(f"\n{entry['id']} ({entry['quant']})")
    print(f"{'check':<18} {'result':<6} {'s':>5}  detail")
    print("-" * 78)
    for row in results:
        print(
            f"{row['check']:<18} {'PASS' if row['passed'] else 'FAIL':<6} "
            f"{row['seconds']:>5}  {row['detail'][:48]}"
        )
    total = score(results)
    print(f"\nscore {total['passed']}/{total['total']} "
          f"({total['score']:.0%}) | peak VRAM {sampler.peak_vram_gib:.1f} GiB")
    _save(args.out, {
        "model": {k: entry[k] for k in ("id", "file", "quant", "repo", "license")},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "score": total,
        "peak_vram_gib": round(sampler.peak_vram_gib, 1),
        "peak_ram_gib": round(sampler.peak_ram_gib, 1),
        "results": results,
    })
    return 0 if total["passed"] == total["total"] else 1


def cmd_doctor(args) -> int:
    registry = _registry(args)
    entry = default_model(registry)
    print(f"python       {sys.version.split()[0]}")
    print(f"repo         {REPO_ROOT}")
    print(f"registry     {args.registry} ({len(registry['models'])} models)")
    print(f"default      {entry['id']} <- {entry['repo']}/{entry['file']}")

    server = find_server_bin()
    print(f"llama-server {server} {'OK' if server.exists() else 'MISSING'}")
    if server.exists():
        try:
            out = subprocess.run(
                [str(server), "--version"], capture_output=True, text=True, timeout=30
            ).stdout.strip().splitlines()
            print(f"             {out[0] if out else '(no version output)'}")
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"             could not run: {exc}")

    cc = ""
    if shutil.which("nvidia-smi"):
        try:
            cc = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap,name,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            cc = ""
    print(f"gpu          {cc or 'none detected (CPU-only: pass --cuda-arch to the fetch script)'}")

    model = Path(args.models_dir) / str(entry["file"])
    state = f"{model.stat().st_size / 1024**3:.2f} GiB" if model.exists() else "not downloaded"
    print(f"weights      {model} ({state})")
    print(f"cache (GCS)  {args.cache_gs or '(disabled; set GGUF_CACHE_GS)'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=REPO_ROOT / "models.json",
        help="model registry (default: models.json)",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=REPO_ROOT / "models",
        help="where GGUF weights are stored (default: ./models)",
    )
    parser.add_argument(
        "--cache-gs", default=DEFAULT_CACHE_GS,
        help="optional GCS mirror, e.g. gs://bucket/gguf-lab/gguf "
             "(default: $GGUF_CACHE_GS)",
    )
    parser.add_argument(
        "--ctx", type=int, default=4096, help="context size (default 4096)",
    )
    parser.add_argument(
        "--ngl", type=int, default=99, help="GPU layers to offload (default 99)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_models = sub.add_parser("models", help="list the registry")
    p_models.add_argument("--json", action="store_true", help="print raw JSON")
    p_models.set_defaults(func=cmd_models, model=None)

    p_pull = sub.add_parser("pull", help="download weights")
    p_pull.add_argument("model", nargs="?", help="model id (default: registry default)")
    p_pull.add_argument("--all", action="store_true", help="download every entry")
    p_pull.set_defaults(func=cmd_pull)

    p_run = sub.add_parser("run", help="one prompt, one answer")
    p_run.add_argument("model", nargs="?")
    p_run.add_argument("--prompt", required=True)
    p_run.add_argument("--system", default=None)
    p_run.add_argument("--max-tokens", type=int, default=256)
    p_run.add_argument("--temperature", type=float, default=0.0)
    p_run.set_defaults(func=cmd_run)

    p_bench = sub.add_parser("bench", help="speed and screening checks")
    bench_sub = p_bench.add_subparsers(dest="bench", required=True)

    p_speed = bench_sub.add_parser("speed", help="prefill/decode throughput")
    p_speed.add_argument("model", nargs="?")
    p_speed.add_argument("--runs", type=int, default=3)
    p_speed.add_argument("--max-tokens", type=int, default=256)
    p_speed.add_argument("--out", type=Path, default=None)
    p_speed.set_defaults(func=cmd_speed)

    p_checks = bench_sub.add_parser("checks", help="deterministic pass/fail checks")
    p_checks.add_argument("model", nargs="?")
    p_checks.add_argument(
        "--only", default=None,
        help="comma-separated: " + ",".join(c.name for c in CHECKS),
    )
    p_checks.add_argument("--system", default=None)
    p_checks.add_argument("--out", type=Path, default=None)
    p_checks.set_defaults(func=cmd_checks)

    p_doctor = sub.add_parser("doctor", help="environment report")
    p_doctor.set_defaults(func=cmd_doctor, model=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
