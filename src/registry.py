"""Loading and validating the GGUF model registry (`models.json`).

The registry is plain data so adding a candidate never means touching code, and
so every entry carries its licence and conversion status next to the file it
points at. Unofficial conversions are marked explicitly: the whole point of this
tool is to measure them, not to trust them.
"""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED = ("id", "repo", "file", "quant", "size_gb", "license")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "models.json"


class RegistryError(ValueError):
    """The registry is malformed or an id cannot be resolved."""


def load_registry(path: Path | str | None = None) -> dict:
    """Read and validate the registry, returning the parsed object."""
    source = Path(path) if path is not None else DEFAULT_REGISTRY
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RegistryError(f"cannot read {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryError(f"{source} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise RegistryError(f"{source} must be an object with a 'models' list")

    seen: set[str] = set()
    for entry in data["models"]:
        if not isinstance(entry, dict):
            raise RegistryError(f"{source}: every model must be an object")
        missing = [key for key in REQUIRED if not entry.get(key)]
        if missing:
            raise RegistryError(
                f"{source}: model {entry.get('id', '?')} missing {missing}"
            )
        if entry["id"] in seen:
            raise RegistryError(f"{source}: duplicate model id {entry['id']}")
        seen.add(entry["id"])
        if float(entry["size_gb"]) <= 0:
            raise RegistryError(f"{source}: {entry['id']} has size_gb <= 0")
        if not str(entry["file"]).endswith(".gguf"):
            raise RegistryError(f"{source}: {entry['id']} file must be .gguf")

    default = data.get("default")
    if default is None:
        data["default"] = data["models"][0]["id"] if data["models"] else None
    elif default not in seen:
        raise RegistryError(f"{source}: default {default!r} is not in models")
    return data


def find_model(registry: dict, model_id: str) -> dict:
    """One model entry by id, or a RegistryError listing the known ids."""
    for entry in registry.get("models", []):
        if entry.get("id") == model_id:
            return entry
    known = ", ".join(entry.get("id", "?") for entry in registry.get("models", []))
    raise RegistryError(f"unknown model id {model_id!r}; known: {known}")


def default_model(registry: dict) -> dict:
    """The entry `models.json` marks as default."""
    model_id = registry.get("default")
    if not model_id:
        raise RegistryError("registry has no default model")
    return find_model(registry, model_id)


def resolve(registry: dict, model_id: str | None) -> dict:
    """`None` means the default model."""
    return default_model(registry) if model_id in (None, "", "default") else find_model(registry, model_id)
