"""Speed numbers must separate prefill from decode and say where they came from."""

from __future__ import annotations

import pytest

from src.speed import one_run, run_speed, summarize


class _TimedServer:
    """Server that reports llama.cpp-style timings."""

    def __init__(self, text: str = "one two three four", prefill_ms: float = 120.0,
                 tok_s: float = 42.0):
        self.text = text
        self.prefill_ms = prefill_ms
        self.tok_s = tok_s
        self.calls = 0

    def start(self):
        return None

    def chat(self, messages, max_tokens=256, **kwargs):
        self.calls += 1
        words = len(self.text.split())
        return {
            "text": self.text,
            "usage": {"prompt_tokens": 20, "completion_tokens": words},
            "timings": {
                "prompt_ms": self.prefill_ms,
                "predicted_per_second": self.tok_s,
                "predicted_n": words,
                "prompt_n": 20,
            },
        }


def test_one_run_uses_server_timings():
    report = one_run(_TimedServer(prefill_ms=250.0, tok_s=37.5))
    assert report["source"] == "timings"
    assert report["prefill_s"] == 0.25
    assert report["decode_tok_s"] == 37.5
    assert report["completion_tokens"] == 4
    assert report["output_words"] == 4


class _NoTimingsServer:
    def start(self):
        return None

    def chat(self, messages, max_tokens=256, **kwargs):
        return {"text": "a b c", "usage": {"completion_tokens": 3}, "timings": {}}


def test_one_run_falls_back_to_wallclock():
    report = one_run(_NoTimingsServer())
    assert report["source"] == "wallclock"
    assert report["prefill_s"] is None
    assert report["decode_tok_s"] is not None
    assert report["e2e_tok_s"] is not None


def test_summarize_reports_range_and_mean():
    runs = [
        {"wall_s": 2.0, "prefill_s": 0.2, "decode_tok_s": 40.0,
         "e2e_tok_s": 20.0, "source": "timings", "completion_tokens": 40},
        {"wall_s": 2.4, "prefill_s": 0.2, "decode_tok_s": 30.0,
         "e2e_tok_s": 16.0, "source": "timings", "completion_tokens": 40},
    ]
    summary = summarize(runs)
    assert summary["mean_decode_tok_s"] == 35.0
    assert summary["min_decode_tok_s"] == 30.0
    assert summary["max_decode_tok_s"] == 40.0
    assert summary["runs"] == 2
    assert summary["source"] == "timings"


def test_run_speed_warms_up_the_server():
    server = _TimedServer()
    report = run_speed(server, runs=3, max_tokens=32)
    assert server.calls == 3
    assert report["runs"] == 3
    assert len(report["raw"]) == 3
    assert report["mean_decode_tok_s"] == 42.0


def test_run_speed_rejects_zero_runs_by_running_once():
    server = _TimedServer()
    report = run_speed(server, runs=0)
    assert report["runs"] == 1
