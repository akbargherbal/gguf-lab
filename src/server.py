"""A tiny llama.cpp server client: one process, reused, stdlib HTTP only.

`llama-server` exposes an OpenAI-compatible endpoint, so the tool sends plain
chat completions and reads the extra `timings` block llama.cpp adds. No SDK, no
torch, no GPU-specific Python: the same code runs on a T4 and on a laptop with a
CPU-only build.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_BIN = REPO_ROOT / "llama.cpp-text" / "build" / "bin" / "llama-server"


class ServerError(RuntimeError):
    """The server could not start, died, or returned nothing usable."""


def build_payload(
    messages: list[dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.0,
    response_format: dict | None = None,
) -> dict:
    """OpenAI-style chat request. Greedy by default for reproducibility."""
    payload: dict[str, object] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    return payload


def parse_response(payload: dict) -> dict:
    """Assistant text plus llama.cpp `usage`/`timings`, or ServerError."""
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ServerError(f"unexpected server response: {payload!r}"[:400]) from exc
    text = ((choice.get("message") or {}).get("content") or "").strip()
    if not text:
        raise ServerError("server returned an empty response")
    return {
        "text": text,
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage") or {},
        "timings": payload.get("timings") or {},
    }


def loader_path_env() -> dict[str, str]:
    """Environment for the child, with CUDA runtime libs on LD_LIBRARY_PATH."""
    env = dict(os.environ)
    helper = REPO_ROOT / "cuda_loader_path.sh"
    if not helper.exists():
        return env
    try:
        out = subprocess.run(
            ["bash", str(helper)], capture_output=True, text=True,
            timeout=60, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return env
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("export LD_LIBRARY_PATH="):
            env["LD_LIBRARY_PATH"] = line.split("=", 1)[1].strip()
    return env


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class LlamaServer:
    """One `llama-server` process serving one GGUF."""

    model_path: Path
    server_bin: Path = DEFAULT_SERVER_BIN
    ctx_size: int = 4096
    n_gpu_layers: int = 99
    threads: int = 0
    port: int = 0
    startup_timeout: float = 300.0
    request_timeout: float = 600.0
    extra_args: tuple[str, ...] = ()
    _proc: object = field(default=None, repr=False, compare=False)
    _port: int = field(default=0, repr=False, compare=False)
    _log_path: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        self.server_bin = Path(self.server_bin)

    def command(self, port: int) -> list[str]:
        args = [
            str(self.server_bin),
            "-m", str(self.model_path),
            "--host", "127.0.0.1",
            "--port", str(port),
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.ctx_size),
            "--jinja",
        ]
        if self.threads > 0:
            args += ["-t", str(self.threads)]
        args.extend(self.extra_args)
        return args

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.server_bin.exists():
            raise ServerError(
                f"llama-server not found at {self.server_bin}; run "
                "./plumbing/fetch_llamacpp.sh"
            )
        if not self.model_path.exists():
            raise ServerError(f"GGUF not found at {self.model_path}; run 'lab.py pull'")

        port = self.port or free_port()
        self._log_path = self.model_path.with_suffix(self.model_path.suffix + ".server.log")
        log = self._log_path.open("w", encoding="utf-8")
        try:
            self._proc = subprocess.Popen(
                self.command(port), stdout=log, stderr=subprocess.STDOUT,
                env=loader_path_env(), start_new_session=True,
            )
        finally:
            log.close()
        self._port = port
        try:
            self._wait_ready(port)
        except ServerError:
            self.stop()
            raise

    def _wait_ready(self, port: int) -> None:
        deadline = time.time() + self.startup_timeout
        url = f"http://127.0.0.1:{port}/health"
        while time.time() < deadline:
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                raise ServerError(
                    f"llama-server exited during startup (code {proc.returncode}): "
                    f"{self.log_tail()}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(0.5)
        raise ServerError(
            f"llama-server not ready in {self.startup_timeout:.0f}s: {self.log_tail()}"
        )

    def log_tail(self, lines: int = 20) -> str:
        if self._log_path is None or not self._log_path.exists():
            return "(no log)"
        text = self._log_path.read_text(encoding="utf-8", errors="replace")
        return " | ".join(text.strip().splitlines()[-lines:])[:1200]

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        response_format: dict | None = None,
    ) -> dict:
        """One stateless completion. Returns text, usage and timings."""
        if not messages:
            raise ServerError("empty message list")
        self.start()
        body = json.dumps(
            build_payload(messages, max_tokens, temperature, response_format)
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self._port}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise ServerError(f"server HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ServerError(f"server request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ServerError(f"server returned invalid JSON: {exc}") from exc
        return parse_response(payload)

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                pass
        except OSError:  # pragma: no cover - already gone
            pass

    # aliases used by the CLI/tests
    unload = stop

    @property
    def log_path(self) -> Path | None:
        return self._log_path


def find_server_bin() -> Path:
    """Locate llama-server: env override, repo build, then PATH."""
    override = os.environ.get("LLAMACPP_SERVER_BIN")
    if override:
        return Path(override)
    if DEFAULT_SERVER_BIN.exists():
        return DEFAULT_SERVER_BIN
    found = shutil.which("llama-server")
    return Path(found) if found else DEFAULT_SERVER_BIN
