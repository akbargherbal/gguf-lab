# Quickstart on a fresh Colab T4

Every step is idempotent; re-running is safe.

```bash
git clone https://github.com/akbargherbal/gguf-lab.git /content/gguf-lab
cd /content/gguf-lab
pip install -q -r requirements.txt
```

## 1. Stage llama.cpp (seconds, no build)

```bash
./plumbing/fetch_llamacpp.sh
llama.cpp-text/build/bin/llama-cli --version
```

On a T4 the arch tag is `sm75-t4`. If the cached binary cannot run on this image
(different CUDA or glibc), the script says so and builds from source instead.

## 2. Environment check

```bash
python scripts/lab.py doctor
```

Confirms GPU compute capability, the server binary, the registry default, and
whether the weights are already downloaded.

## 3. Get the default model

```bash
python scripts/lab.py pull
```

`Qwen3-4B-Instruct-2507-Q8_0.gguf` is ~4.3 GB. To reuse a GCS mirror:

```bash
export GGUF_CACHE_GS=gs://akbar-december-2024-backup/music_captioning/gguf
python scripts/lab.py pull
```

The mirror is keyed by the registry `id`:

```
$GGUF_CACHE_GS/<id>/<file>
```

## 4. Measure

```bash
python scripts/lab.py run --prompt "Name three uses for a haiku."
python scripts/lab.py bench speed --runs 3 --out results/speed.json
python scripts/lab.py bench checks --out results/checks.json
```

## 5. Add a candidate

Edit `models.json`:

```json
{
  "id": "my-model-q4km",
  "label": "My Model (Q4_K_M)",
  "repo": "somebody/My-Model-GGUF",
  "file": "My-Model-Q4_K_M.gguf",
  "quant": "Q4_K_M",
  "size_gb": 2.6,
  "vram_gb": 3.6,
  "license": "Apache-2.0",
  "official_conversion": false,
  "notes": "why it is interesting"
}
```

Then:

```bash
python scripts/lab.py pull my-model-q4km
python scripts/lab.py bench checks my-model-q4km --out results/my-model.json
```

Never commit the weights: `models/` and `*.gguf` are ignored.

## 6. Restarting a dead runtime

Colab loses `/content`. Re-run steps 1–3; results live in git only if you copy
them into the repo (they are small JSON files, and `results/` is ignored by
default — commit deliberately if you want to publish a comparison).
