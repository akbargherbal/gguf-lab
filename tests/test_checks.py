"""The screening checks must pass good answers and reject bad ones.

These are deterministic evaluators: no model, no judge, no GPU. If a check cannot
tell a correct answer from a broken one, it is worse than no check.
"""

from __future__ import annotations

import json

import pytest

from src.checks import (
    CHECKS,
    distinct_ngram_ratio,
    eval_arithmetic,
    eval_context_recall,
    eval_exact_ready,
    eval_json_object,
    eval_no_repetition,
    eval_not_happened,
    eval_three_bullets,
    find_check,
    run_checks,
    score,
)


def test_every_check_has_a_unique_name():
    names = [c.name for c in CHECKS]
    assert len(names) == len(set(names))
    for check in CHECKS:
        assert check.prompt and check.max_tokens > 0 and check.goal


def test_exact_reply_normalises_punctuation():
    assert eval_exact_ready("READY")[0]
    assert eval_exact_ready("  ready.  ")[0]
    assert eval_exact_ready('"READY"')[0]
    assert not eval_exact_ready("Ready to help")[0]


def test_json_object_accepts_fenced_and_plain():
    good = json.dumps({"title": "T", "year": 1999, "tags": ["a", "b"]})
    assert eval_json_object(good)[0]
    assert eval_json_object(f"```json\n{good}\n```")[0]


@pytest.mark.parametrize("bad", [
    "not json",
    '{"title": "T", "year": "1999", "tags": ["a", "b"]}',
    '{"title": "T", "year": 1999, "tags": ["a"]}',
    '{"title": "T", "year": 1999, "tags": ["a", "b"], "extra": 1}',
    '["a", "b"]',
])
def test_json_object_rejects_bad_shapes(bad):
    assert not eval_json_object(bad)[0]


def test_arithmetic():
    assert eval_arithmetic("91")[0]
    assert eval_arithmetic("91 pens are left.")[0]
    assert not eval_arithmetic("96")[0]
    assert not eval_arithmetic("no idea")[0]


def test_three_bullets():
    good = "- better mood\n- stronger heart\n- more energy"
    assert eval_three_bullets(good)[0]
    assert not eval_three_bullets("- one\n- two")[0]
    assert not eval_three_bullets(good + "\n- extra")[0]
    long_line = "- " + " ".join(["word"] * 9)
    assert not eval_three_bullets(f"{long_line}\n- b\n- c")[0]


def test_context_recall_accepts_separators():
    assert eval_context_recall("4719")[0]
    assert eval_context_recall("4-7-1-9")[0]
    assert not eval_context_recall("4718")[0]


def test_unknown_future():
    assert eval_not_happened("NOT HAPPENED")[0]
    assert eval_not_happened("That has not happened yet.")[0]
    assert not eval_not_happened("Brazil won the 2029 World Cup.")[0]


MOUNTAIN = (
    "Granite peaks rise sharply above the valley, their ridgelines cut by wind "
    "and frost. Lower slopes carry pine forest, thinning as altitude climbs "
    "toward bare scree. A glacial stream threads between boulders, cold and "
    "fast, fed by snowfields that persist through summer. Climbers follow a "
    "narrow path along the eastern shoulder, where exposure demands care and "
    "steady footing. Near the summit, clouds often close in without warning, "
    "obscuring cairns and forcing retreat. Descending at dusk, the temperature "
    "drops quickly; headlamps catch reflective markers nailed into rock, "
    "guiding tired legs back toward the trailhead."
)


def test_repetition_metric():
    looping = "the same sentence over and over " * 20
    assert distinct_ngram_ratio(looping) < 0.55
    assert distinct_ngram_ratio(MOUNTAIN) > 0.55


def test_no_repetition_checks_length_and_looping():
    assert eval_no_repetition(MOUNTAIN)[0]
    assert not eval_no_repetition("too short")[0]
    assert not eval_no_repetition("the same thing again and again " * 30)[0]


class _FakeServer:
    def __init__(self, replies: dict[str, str]):
        self.replies = replies
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages, max_tokens=256, **kwargs):
        self.calls.append(messages)
        prompt = messages[-1]["content"]
        for check in CHECKS:
            if check.prompt == prompt:
                return {"text": self.replies.get(check.name, "?"), "usage": {},
                        "timings": {}, "finish_reason": "stop"}
        return {"text": "?", "usage": {}, "timings": {}, "finish_reason": "stop"}


def test_run_checks_collects_and_scores():
    server = _FakeServer({
        "exact_reply": "READY",
        "arithmetic": "91",
        "json_object": '{"title": "T", "year": 1999, "tags": ["a", "b"]}',
    })
    results = run_checks(server, only=["exact_reply", "arithmetic", "json_object"])
    assert len(results) == 3
    assert score(results) == {"passed": 3, "total": 3, "score": 1.0}
    assert all("seconds" in r and "output" in r for r in results)


def test_run_checks_survives_a_crashing_server():
    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("server died")

    results = run_checks(Boom(), only=["exact_reply"])
    assert results[0]["passed"] is False
    assert "server died" in results[0]["detail"]
    assert score(results)["score"] == 0.0


def test_find_check_unknown_lists_names():
    with pytest.raises(KeyError, match="known:"):
        find_check("nope")
