"""Peak memory sampling for a benchmark run.

A T4 has 15 GiB of VRAM and Colab gives 12 or 52 GiB of RAM. Throughput is a GPU
number, but *fit* is not: a model that spills layers to host memory looks fast in
tokens/second until you notice it used 20 GiB of RAM. Sampling both keeps the
report honest and is cheap (two reads every couple of seconds).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path


class MemorySampler:
    """Thread that records peak system RAM and GPU VRAM in GiB."""

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self.peak_ram_gib = 0.0
        self.peak_vram_gib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def ram_used_gib() -> float:
        try:
            values: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, _, rest = line.partition(":")
                values[key] = int(rest.split()[0])  # kB
            total = values.get("MemTotal", 0) / 1024**2
            available = values.get("MemAvailable", 0) / 1024**2
            return max(0.0, total - available)
        except (OSError, ValueError, IndexError):
            return 0.0

    @staticmethod
    def vram_used_gib() -> float:
        if shutil.which("nvidia-smi") is None:
            return 0.0
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout
            used = [
                float(part) for part in out.split()
                if part.replace(".", "", 1).isdigit()
            ]
            return max(used) / 1024 if used else 0.0
        except (OSError, subprocess.SubprocessError, ValueError):
            return 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_ram_gib = max(self.peak_ram_gib, self.ram_used_gib())
            self.peak_vram_gib = max(self.peak_vram_gib, self.vram_used_gib())
            self._stop.wait(self.interval)

    def __enter__(self) -> "MemorySampler":
        self._thread = threading.Thread(
            target=self._run, name="mem-sampler", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)


def percentile(values: list[float], fraction: float) -> float:
    """Simple nearest-rank percentile; enough for a handful of runs."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]
