"""Fetch GGUF weights from Hugging Face, with an optional GCS mirror.

Two jobs, both idempotent:

* if the file is already in ``models/``, do nothing;
* otherwise download it (Hugging Face) and, when a mirror is configured, copy it
  from/to Google Cloud Storage so a Colab reset does not cost another 4 GB.

Third-party conversions can change under the same filename. Treat the first
download as the reference and compare outputs before adopting a new revision;
`models.json` records the repo, not a commit, on purpose (a pinned HF revision
would silently stop tracking security fixes).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


class DownloadError(RuntimeError):
    """Weights could not be fetched."""


def model_path(entry: dict, models_dir: Path | str = DEFAULT_MODELS_DIR) -> Path:
    """Where a registry entry lives on disk."""
    return Path(models_dir) / str(entry["file"])


def _gcloud() -> str | None:
    return shutil.which("gcloud") or shutil.which("gsutil")


def _gcs_copy(source: str, dest: Path) -> bool:
    """Best-effort download through gcloud/gsutil."""
    cli = _gcloud()
    if cli is None:
        return False
    command = "storage cp" if Path(cli).name == "gcloud" else "cp"
    args = [cli, *command.split(), source, str(dest)]
    try:
        return subprocess.run(args, capture_output=True, timeout=1800).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _gcs_upload(source: Path, dest: str) -> bool:
    cli = _gcloud()
    if cli is None:
        return False
    command = "storage cp" if Path(cli).name == "gcloud" else "cp"
    args = [cli, *command.split(), str(source), dest]
    try:
        return subprocess.run(args, capture_output=True, timeout=1800).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def ensure_model(
    entry: dict,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    cache_gs: str | None = None,
    log=print,
) -> Path:
    """Return a local path to the GGUF, downloading it if needed."""
    target = model_path(entry, models_dir)
    if target.exists() and target.stat().st_size > 0:
        log(f"[models] {entry['id']}: already present ({target})")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    if cache_gs:
        remote = f"{cache_gs.rstrip('/')}/{entry['id']}/{entry['file']}"
        log(f"[models] {entry['id']}: trying cache {remote}")
        if _gcs_copy(remote, target):
            log(f"[models] {entry['id']}: restored from cache")
            return target
        log(f"[models] {entry['id']}: cache miss")

    log(f"[models] {entry['id']}: downloading {entry['repo']}/{entry['file']}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - environment issue
        raise DownloadError(
            "huggingface_hub is not installed; run 'pip install -r requirements.txt'"
        ) from exc

    try:
        hf_hub_download(
            repo_id=str(entry["repo"]),
            filename=str(entry["file"]),
            local_dir=str(Path(models_dir)),
        )
    except Exception as exc:  # noqa: BLE001 - surfaced with context
        raise DownloadError(
            f"could not download {entry['repo']}/{entry['file']}: {exc}"
        ) from exc

    if not target.exists():
        raise DownloadError(f"download finished but {target} is missing")

    if cache_gs:
        remote = f"{cache_gs.rstrip('/')}/{entry['id']}/{entry['file']}"
        if _gcs_upload(target, remote):
            log(f"[models] {entry['id']}: mirrored to {remote}")
        else:
            log(f"[models] {entry['id']}: WARNING could not mirror to GCS")
    return target
