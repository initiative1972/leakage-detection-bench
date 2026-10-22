# ADR 0001 — Deterministic grader; no live LLM in CI

## Status
Accepted.

## Context
LeakBench is meant to measure leakage-detection judgement, including that of
LLMs. But a benchmark is only trustworthy if its *own* harness is reproducible.
Calling a hosted model in CI would make results non-deterministic, require
secrets, incur cost, and break offline reproduction.

## Decision
- The grader is **pure, deterministic standard-library code** with no model and
  no randomness. Scores are a function of (items, predictions) only.
- LLMs are supported through an **adapter interface** (`llm_adapter.LLMJudge`)
  that takes a user-supplied `complete(prompt)->str`. Running a live model is an
  explicit *you-run-it* step; **no network call lives in the repo**.
- A `ScriptedJudge` test double exercises the adapter end-to-end deterministically.

## Consequences
- CI is offline, fast, secret-free, and green by construction.
- The published leaderboard contains only reproducible non-LLM baselines; LLM
  rows are produced by whoever runs a model, with their own pinned prompt.
- Honest "runs vs design" boundary, consistent across this portfolio.
