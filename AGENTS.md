# AGENTS.md

Instructions for AI coding agents working in this repo.

## What this is

A screening harness for small GGUF models (often unofficial conversions) on a
CUDA GPU, defaulting to a size that fits a 15 GiB T4. It answers: does it run,
does it follow instructions, how fast is it, and how much memory does it use.

The llama.cpp binary is **not** built here during normal use. It is fetched from
an arch-tagged cache in GCS (see `plumbing/`), verified by sha256, and smoke
tested; a source build is only the fallback.

## Hard constraints

- Target GPU: T4, compute capability 7.5 (`sm_75`), 15 GiB VRAM, no bf16, no
  FlashAttention. Never enable bf16.
- Colab runtimes are ephemeral: anything outside the repo (and outside a mounted
  Drive) is lost. Every step must be idempotent and re-runnable.
- Weights (`models/`, `*.gguf`) and binaries (`llama.cpp-text/`) are never
  committed.
- Python only for glue. No torch requirement: the client talks HTTP to
  `llama-server` with the standard library.
- Do not add a dependency unless it removes real code. Currently:
  `huggingface_hub` (downloads) and `pytest` (tests).

## Conventions

- `models.json` is the only place a model is declared. Fields: `id`, `label`,
  `repo`, `file`, `quant`, `size_gb`, `vram_gb`, `license`,
  `official_conversion`, `notes`. Mark unofficial conversions honestly.
- The default model must fit a T4 alone with headroom; a test enforces
  `vram_gb <= 15`.
- Checks in `src/checks.py` are **deterministic**: code decides pass/fail. Do not
  introduce an LLM judge, and do not soften a check to make a model pass.
- Every check has a `goal` string so a report is readable without the source.
- Speed numbers must state their `source`: `timings` (from llama.cpp) or
  `wallclock`. Never mix the two in one comparison.
- Server client: stdlib `urllib` only, greedy by default (`temperature: 0`),
  one fresh request per prompt (no conversation state).
- Startup failures must say what to run: `./plumbing/fetch_llamacpp.sh` for the
  binary, `lab.py pull` for the weights.

## Testing

- `pytest tests/` must pass on CPU with no model present and no network.
- Test cases to keep: registry validation (duplicate ids, missing fields,
  default must exist), each check's accept **and** reject path, wall-clock
  fallback when `timings` are missing, and actionable errors for a missing
  binary/GGUF.
- A real GPU run is a manual step, never part of `pytest`.

## Reporting results

A result is only meaningful with the model id, quant, file, and date. Prefer
`--out results/<model>.json`; the file records the registry entry and the raw
per-run numbers. Do not publish a comparison that mixes quants or engines without
saying so.

## Guardrails

- Do not change the default model, the check set, or the scoring rule to make a
  particular candidate look better.
- Respect licences: most entries are third-party conversions. `license` in
  `models.json` must match the upstream repo.
- Keep attributions: llama.cpp is MIT (ggml-org); each model keeps its own
  licence.
