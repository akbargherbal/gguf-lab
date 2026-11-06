"""Download helpers: presence check, cache path shape, no network in tests."""

from __future__ import annotations

from pathlib import Path

import src.download as download


ENTRY = {
    "id": "demo",
    "repo": "org/model-GGUF",
    "file": "model-Q8_0.gguf",
}


def test_existing_file_is_returned_without_downloading(tmp_path, monkeypatch):
    target = tmp_path / ENTRY["file"]
    target.write_bytes(b"weights")

    def boom(*args, **kwargs):
        raise AssertionError("must not download")

    monkeypatch.setattr(download, "_gcs_copy", boom)
    assert download.ensure_model(ENTRY, tmp_path, cache_gs="gs://b/p") == target


def test_cache_hit_skips_hugging_face(tmp_path, monkeypatch):
    calls: list[tuple[str, Path]] = []

    def fake_gcs(source: str, dest: Path) -> bool:
        calls.append((source, dest))
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"cached")
        return True

    monkeypatch.setattr(download, "_gcs_copy", fake_gcs)
    monkeypatch.setattr(download, "_gcs_upload", lambda *a, **k: True)

    path = download.ensure_model(ENTRY, tmp_path, cache_gs="gs://b/p", log=lambda *_: None)

    assert path.read_bytes() == b"cached"
    assert calls[0][0] == "gs://b/p/demo/model-Q8_0.gguf"


def test_cache_miss_without_huggingface_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "_gcs_copy", lambda *a, **k: False)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", None)

    try:
        download.ensure_model(ENTRY, tmp_path, cache_gs="gs://b/p", log=lambda *_: None)
    except download.DownloadError as exc:
        assert "huggingface_hub" in str(exc)
    else:  # pragma: no cover - the import must fail
        raise AssertionError("expected DownloadError")


def test_model_path_uses_registry_filename(tmp_path):
    assert download.model_path(ENTRY, tmp_path) == tmp_path / "model-Q8_0.gguf"
