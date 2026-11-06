# The checks

Seven deterministic checks. Each returns pass/fail and a one-line reason, so a
report can be compared across models without a judge model in the loop.

| check | what it catches | pass rule |
|---|---|---|
| `exact_reply` | chat template / stop-sequence breakage, instruction ignoring | output is exactly `READY` after trimming quotes and punctuation |
| `json_object` | broken structured output | parses as a JSON object with keys `title` (str), `year` (int), `tags` (2 strings) |
| `arithmetic` | tokenizer damage, lost reasoning on small quants | first number in the reply is `91` |
| `three_bullets` | failure to respect structure and length | exactly 3 lines starting with `- `, each ≤ 8 words |
| `context_recall` | context window or RoPE problems | reply contains `4719` from a needle inside ~900 words of filler |
| `unknown_future` | confident fabrication | hedges about the 2029 World Cup (`has not happened`, `not yet`, …) |
| `no_repetition` | quant damage that makes the model loop | 60–160 words and distinct-bigram ratio ≥ 0.55 |

## Why these, and not prose quality

Prose quality needs either a human or a judge model; both are slow, expensive,
and unstable across runs. The point here is to reject candidates cheaply *before*
spending that effort. A model that passes 7/7 is a candidate for real evaluation,
not a winner.

## Reading a score

- **7/7** — behaves like a functioning instruct model at this size.
- **5–6/7** — usually one specific weakness (JSON, or long context). Look at which
  check failed before rejecting it for your task.
- **≤ 4/7** — likely a broken conversion or a base model without an instruct
  template. Check the chat template before blaming quantization.

## Reproducibility

All checks use `temperature: 0`, so the same weights and binary give the same
answers. If two runs differ on the same file, suspect a different `llama.cpp`
build or a changed GGUF (re-upload), not sampling.

## Adding a check

Add a `Check(...)` entry to `CHECKS` with a unique `name`, a `goal`, a prompt, and
an evaluator that returns `(passed, detail)`. Add both an accept and a reject case
to `tests/test_checks.py`. Do not add a check that depends on wall-clock timing or
network access.
