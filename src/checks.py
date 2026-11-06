"""Deterministic screening checks: pass/fail, not an LLM judge.

An unofficial GGUF can be broken in boring ways (chat template missing, quant
damage, wrong tokenizer, truncated context). These checks catch those cheaply and
reproducibly: exact answers, valid JSON, bullet count, a needle in a long
context, a refusal, and repetition. Each one is scored by code, so two runs of
the same weights agree; nothing here is a measure of prose quality.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Callable

Evaluator = Callable[[str], tuple[bool, str]]


@dataclass(frozen=True)
class Check:
    name: str
    goal: str
    prompt: str
    max_tokens: int
    evaluate: Evaluator


def _normalize(text: str) -> str:
    return text.strip().strip("\"'`*_ .!\n\t").lower()


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def eval_exact_ready(text: str) -> tuple[bool, str]:
    got = _normalize(text)
    return got == "ready", f"got {got[:40]!r}"


def eval_json_object(text: str) -> tuple[bool, str]:
    raw = _strip_fence(text)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, f"not JSON: {exc}"
    if not isinstance(obj, dict):
        return False, f"got {type(obj).__name__}, want object"
    if set(obj) != {"title", "year", "tags"}:
        return False, f"keys {sorted(obj)}"
    if not isinstance(obj.get("title"), str):
        return False, "title is not a string"
    if not isinstance(obj.get("year"), int) or isinstance(obj.get("year"), bool):
        return False, "year is not an integer"
    tags = obj.get("tags")
    if not isinstance(tags, list) or len(tags) != 2 or not all(isinstance(t, str) for t in tags):
        return False, "tags is not a list of two strings"
    return True, "valid shape"


def eval_arithmetic(text: str) -> tuple[bool, str]:
    numbers = re.findall(r"-?\d+", text.replace(",", ""))
    if not numbers:
        return False, "no number in the answer"
    if numbers[0] != "91":
        return False, f"got {numbers[0]}, want 91"
    return True, "91"


def eval_three_bullets(text: str) -> tuple[bool, str]:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    bullets = [ln for ln in lines if re.match(r"^[-*•]\s+", ln)]
    if len(bullets) != 3:
        return False, f"{len(bullets)} bullet lines, want 3"
    for line in bullets:
        body = re.sub(r"^[-*•]\s+", "", line)
        if len(body.split()) > 8:
            return False, f"bullet too long: {body[:40]!r}"
    return True, "3 bullets, each <= 8 words"


def eval_context_recall(text: str) -> tuple[bool, str]:
    digits = re.sub(r"\D", "", text)
    return digits.startswith("4719"), f"got {text.strip()[:40]!r}"


def eval_not_happened(text: str) -> tuple[bool, str]:
    low = text.lower()
    patterns = (
        "not happened", "hasn't happened", "has not happened",
        "didn't happen", "did not happen", "not yet", "future", "hasn't occurred",
        "has not occurred", "not occurred",
    )
    ok = any(p in low for p in patterns)
    return ok, ("hedged correctly" if ok else f"asserted: {text.strip()[:60]!r}")


def distinct_ngram_ratio(text: str, n: int = 2) -> float:
    words = re.findall(r"[a-z0-9']+", text.lower())
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def eval_no_repetition(text: str) -> tuple[bool, str]:
    words = text.split()
    if not 60 <= len(words) <= 160:
        return False, f"{len(words)} words, want 60-160"
    ratio = distinct_ngram_ratio(text, 2)
    if ratio < 0.55:
        return False, f"distinct-2 ratio {ratio:.2f} < 0.55 (looping)"
    return True, f"distinct-2 {ratio:.2f}"


_NEEDLE_FILLER = " ".join(
    f"Entry {i:02d} records a routine note about weather, transport, and supplies."
    for i in range(1, 61)
)


CHECKS: tuple[Check, ...] = (
    Check(
        name="exact_reply",
        goal="obey a literal format constraint",
        prompt="Reply with exactly the word READY and nothing else. "
               "No punctuation, no quotes, no explanation.",
        max_tokens=16,
        evaluate=eval_exact_ready,
    ),
    Check(
        name="json_object",
        goal="emit syntactically valid JSON with the requested shape",
        prompt='Return a JSON object with exactly three keys: "title" (string), '
               '"year" (integer), "tags" (array of exactly two strings). '
               "Output only the JSON.",
        max_tokens=128,
        evaluate=eval_json_object,
    ),
    Check(
        name="arithmetic",
        goal="follow a two-step word problem",
        prompt="A crate holds 12 boxes. Each box has 8 pens. 5 pens are removed. "
               "How many pens are left? Reply with the number only.",
        max_tokens=16,
        evaluate=eval_arithmetic,
    ),
    Check(
        name="three_bullets",
        goal="respect structure and a length limit per item",
        prompt="List exactly three benefits of walking, as three lines starting "
               'with "- ". Each line must be at most 8 words.',
        max_tokens=96,
        evaluate=eval_three_bullets,
    ),
    Check(
        name="context_recall",
        goal="retrieve a fact from a long prompt (context handling)",
        prompt=(
            _NEEDLE_FILLER
            + " Archive note: the vault code is 4-7-1-9 and it does not change. "
            + _NEEDLE_FILLER
            + " What is the vault code? Reply with the four digits only."
        ),
        max_tokens=16,
        evaluate=eval_context_recall,
    ),
    Check(
        name="unknown_future",
        goal="refuse to invent an event that has not happened",
        prompt="Who won the 2029 FIFA World Cup? If that event has not happened, "
               "reply exactly NOT HAPPENED.",
        max_tokens=32,
        evaluate=eval_not_happened,
    ),
    Check(
        name="no_repetition",
        goal="produce 60-160 words without looping",
        prompt="Describe a mountain in 100 words. Do not repeat sentences or "
               "phrases.",
        max_tokens=220,
        evaluate=eval_no_repetition,
    ),
)


def find_check(name: str) -> Check:
    for check in CHECKS:
        if check.name == name:
            return check
    known = ", ".join(c.name for c in CHECKS)
    raise KeyError(f"unknown check {name!r}; known: {known}")


def run_checks(server, only: list[str] | None = None, system: str | None = None) -> list[dict]:
    """Run checks against a started server; returns one result dict each."""
    selected = [find_check(name) for name in only] if only else list(CHECKS)
    results: list[dict] = []
    for check in selected:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": check.prompt})
        started = time.time()
        try:
            reply = server.chat(messages, max_tokens=check.max_tokens)
            text = reply["text"]
            passed, detail = check.evaluate(text)
        except Exception as exc:  # noqa: BLE001 - one broken check must not stop the rest
            text, passed, detail = "", False, f"error: {type(exc).__name__}: {exc}"
        results.append({
            "check": check.name,
            "goal": check.goal,
            "passed": passed,
            "detail": detail,
            "seconds": round(time.time() - started, 2),
            "output_words": len(text.split()),
            "output": text[:2000],
        })
    return results


def score(results: list[dict]) -> dict:
    passed = sum(1 for r in results if r["passed"])
    return {
        "passed": passed,
        "total": len(results),
        "score": round(passed / len(results), 3) if results else 0.0,
    }
