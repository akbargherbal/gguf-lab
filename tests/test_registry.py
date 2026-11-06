"""The registry is data: validate it, never guess at a missing default."""

from __future__ import annotations

import json

import pytest

from src.registry import (
    RegistryError,
    default_model,
    find_model,
    load_registry,
    resolve,
)


def _write(tmp_path, data):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_repo_registry_is_valid():
    registry = load_registry()
    assert registry["models"]
    entry = default_model(registry)
    assert entry["file"].endswith(".gguf")
    assert entry["size_gb"] > 0


def test_default_model_fits_a_t4():
    entry = default_model(load_registry())
    assert entry["vram_gb"] <= 15, "default must fit a 15 GiB T4"


def test_duplicate_ids_rejected(tmp_path):
    entry = {
        "id": "x", "repo": "r", "file": "m.gguf", "quant": "Q8_0",
        "size_gb": 1, "license": "MIT",
    }
    path = _write(tmp_path, {"default": "x", "models": [entry, dict(entry)]})
    with pytest.raises(RegistryError, match="duplicate"):
        load_registry(path)


def test_missing_field_rejected(tmp_path):
    path = _write(tmp_path, {"default": "x", "models": [
        {"id": "x", "repo": "r", "file": "m.gguf", "quant": "Q8_0"},
    ]})
    with pytest.raises(RegistryError, match="missing"):
        load_registry(path)


def test_default_must_exist(tmp_path):
    path = _write(tmp_path, {"default": "nope", "models": [
        {"id": "x", "repo": "r", "file": "m.gguf", "quant": "Q8_0",
         "size_gb": 1, "license": "MIT"},
    ]})
    with pytest.raises(RegistryError, match="not in models"):
        load_registry(path)


def test_non_gguf_rejected(tmp_path):
    path = _write(tmp_path, {"default": "x", "models": [
        {"id": "x", "repo": "r", "file": "m.bin", "quant": "Q8_0",
         "size_gb": 1, "license": "MIT"},
    ]})
    with pytest.raises(RegistryError, match=".gguf"):
        load_registry(path)


def test_resolve_default_and_by_id():
    registry = load_registry()
    assert resolve(registry, None)["id"] == registry["default"]
    known = registry["models"][0]["id"]
    assert resolve(registry, known)["id"] == known
    with pytest.raises(RegistryError, match="unknown model id"):
        find_model(registry, "does-not-exist")
