# gguf-lab

Screen small GGUF models — mostly unofficial conversions — on real hardware,
before you trust one in a project.

Small quantized models are cheap to download and easy to run, which is exactly
why it is easy to adopt a broken one: a missing chat template, a damaged quant, a
tokenizer mismatch, or a model that quietly spills to system RAM. `gguf-lab`
answers three questions with numbers instead of vibes:

1. **Does it run?** one command, one `llama-server`, clear errors.
2. **Does it obey?** seven deterministic checks (JSON shape, exact format, a
   long-context needle, refusal, repetition) scored by code, not by a judge.
3. **How fast?** prefill and decode tokens/second, plus peak VRAM/RAM.

Default model: **Qwen3-4B-Instruct-2507 Q8_0** (~4.3 GB) — runs alone on a 15 GiB
T4 with room to spare. Change it in [`models.json`](models.json); no code edits.

## Quickstart (Colab T4 or any CUDA box)

```bash
git clone https://github.com/akbargherbal/gguf-lab.git
cd gguf-lab
pip install -q -r requirements.txt

./plumbing/fetch_llamacpp.sh     # prebuilt, arch-matched (sm75-t4 on a T4)
python scripts/lab.py doctor
python scripts/lab.py pull       # default model -> models/
```

Then measure:

```bash
python scripts/lab.py run --prompt "Explain what a cello is in one sentence."

python scripts/lab.py bench speed
#   wall 6.1s | prefill 0.4s | decode 24.8 tok/s (24.1-25.2)
#   peak RAM 3.9 GiB | peak VRAM 5.4 GiB | source timings

python scripts/lab.py bench checks --out results/qwen3-q8.json
#   check              result     s  detail
#   exact_reply        PASS    0.6  ready
#   json_object        PASS    1.1  valid shape
#   ...
#   score 7/7 (100%)
```

Compare two models:

```bash
python scripts/lab.py pull qwen3-4b-instruct-2507-q4km
python scripts/lab.py bench checks qwen3-4b-instruct-2507-q4km --out results/q4.json
python scripts/lab.py bench speed  qwen3-4b-instruct-2507-q4km --out results/q4-speed.json
```

## The binary cache (no 20-minute builds)

`plumbing/fetch_llamacpp.sh` downloads an arch-tagged tarball instead of
compiling. The default source is the cache already used by
[`music_captioning`](https://github.com/akbargherbal/music_captioning):

```
gs://akbar-december-2024-backup/music_captioning/binaries/sm75-t4/
  llamacpp-text.tar.gz
  llamacpp-text.tar.gz.sha256
  manifest.json      # llama.cpp commit, CUDA toolkit, flags, size, sha256
```

The fetch verifies the checksum, extracts, loads `cuda_loader_path.sh`, and
smoke-tests `llama-cli --version`; a stale or incompatible entry is rejected and
the script builds from source instead.

To use your own prefix:

```bash
export BINARIES_GS=gs://your-bucket/llamacpp/binaries
./plumbing/fetch_llamacpp.sh --build --publish --cuda-arch 75   # CPU VM, no GPU needed
./plumbing/fetch_llamacpp.sh                                    # later: fetch
```

## Model registry

| id | quant | ~GB | VRAM | conversion |
|---|---|---:|---:|---|
| `qwen3-4b-instruct-2507-q8` (default) | Q8_0 | 4.3 | 5.5 | unofficial |
| `qwen3-4b-instruct-2507-q4km` | Q4_K_M | 2.5 | 3.5 | unofficial |
| `qwen3-8b-q4km` | Q4_K_M | 4.7 | 6.5 | unofficial |
| `llama32-3b-instruct-q8` | Q8_0 | 3.2 | 4.5 | unofficial |
| `granite-4.1-3b-q8` | Q8_0 | 3.2 | 4.5 | **official (IBM)** |

Every entry carries `license`, `official_conversion`, and `size_gb` so a result
can be read without opening another tab.

## What this is not

- **Not a leaderboard.** Seven checks catch breakage, not prose quality; a model
  can pass all of them and still be dull. Use them to reject, not to crown.
- **Not a judge.** No second LLM scores the answers; every check is code.
- **Not a guarantee.** An unofficial GGUF can be fine today and re-uploaded
  tomorrow. Keep the file you measured, and record its sha256 when it matters.

## Layout

```
models.json              registry (default model, licences, sizes)
plumbing/                binary cache: env, arch tag, pin, fetch/build/publish
cuda_loader_path.sh      CUDA runtime path fix for cached binaries
src/registry.py          validate/select models
src/download.py          HF download + optional GCS mirror
src/server.py            llama-server lifecycle + OpenAI-compatible chat
src/checks.py            deterministic screening checks
src/speed.py             prefill/decode throughput
src/metrics.py           peak RAM/VRAM sampling
scripts/lab.py           CLI
tests/                   pure-Python tests (no GPU, no weights)
```

## Tests

```bash
python -m pytest tests/ -q     # no model, no GPU needed
```

## Licence

MIT for this tool. Model weights keep their own licences — see the `license`
column in `models.json` and the upstream repo before any commercial use.
