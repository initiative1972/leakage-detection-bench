"""LLM-judge adapter -- *interface only*; no network calls live in this repo.

This is the seam for plugging a real model into LeakBench. You provide a
``complete(prompt: str) -> str`` callable (OpenAI, Anthropic, a local server,
whatever); the adapter builds a strict prompt, parses the JSON answer, and
returns a ``Prediction``. Running a live model is a *you-run-it* step, so CI
stays offline, deterministic and secret-free -- consistent with the "runs vs
design" boundary used across this portfolio.

A deterministic ``ScriptedCompletion`` test double is provided so the parsing
and plumbing can be unit-tested without any model.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict

from .config import LEAK_TYPES
from .schema import Prediction, TaskSpec

PROMPT_TEMPLATE = """You are auditing a machine-learning task specification for \
DATA LEAKAGE.

Leakage means a feature or setup lets information that would NOT be available at \
prediction time (or that encodes the label) influence the model.

Taxonomy (choose exactly one leak_type, or "none"):
- target_leakage: a feature is a consequence/proxy of the label.
- temporal_leakage: a feature uses information from after the prediction time.
- preprocessing_leakage: a transform is fit on all data before the split.
- group_leakage: the same entity appears in both train and test.
- proxy_leakage: an id/ordering/source encodes the label.
- none: the spec is well-posed.

Specification:
{spec}

Respond with ONLY a JSON object:
{{"has_leakage": <true|false>, "leak_type": "<one of the taxonomy>", \
"offending_features": ["<feature name>", ...], "rationale": "<one sentence>"}}
"""


def build_prompt(spec: TaskSpec) -> str:
    return PROMPT_TEMPLATE.format(spec=spec.public_text())


def parse_response(text: str) -> Prediction:
    """Parse a model's JSON answer into a Prediction (tolerant of extra prose)."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model response")
    obj = json.loads(match.group(0))
    leak_type = obj.get("leak_type", "none")
    if leak_type not in LEAK_TYPES:
        leak_type = "none" if not obj.get("has_leakage") else "target_leakage"
    return Prediction(
        has_leakage=bool(obj.get("has_leakage", False)),
        leak_type=leak_type,
        offending_features=list(obj.get("offending_features", []) or []),
    )


class LLMJudge:
    """Wrap a ``complete(prompt)->str`` callable into a LeakBench solver."""

    def __init__(self, complete: Callable[[str], str], name: str = "llm_judge"):
        self._complete = complete
        self.name = name

    def __call__(self, spec: TaskSpec) -> Prediction:
        raw = self._complete(build_prompt(spec))
        return parse_response(raw)


class ScriptedCompletion:
    """A deterministic ``complete`` stand-in for tests: maps spec.id -> raw JSON.

    Unknown ids fall back to a 'no leakage' answer. This exercises the adapter
    end-to-end with zero dependencies and no network.
    """

    def __init__(self, by_id: Dict[str, str]):
        self._by_id = by_id
        self._last_spec_id = None

    def for_spec(self, spec: TaskSpec) -> str:
        return self._by_id.get(
            spec.id,
            '{"has_leakage": false, "leak_type": "none", '
            '"offending_features": [], "rationale": "looks fine"}',
        )

    def __call__(self, prompt: str) -> str:  # pragma: no cover - see note
        # Not used directly; tests drive via `for_spec`. Kept so the type is a
        # valid `complete` callable if a spec id is embedded in the prompt.
        raise NotImplementedError(
            "ScriptedCompletion is spec-addressed; use ScriptedJudge in tests."
        )


class ScriptedJudge:
    """Test-only solver backed by canned per-id responses (deterministic)."""

    def __init__(self, by_id: Dict[str, str], name: str = "scripted_judge"):
        self._scripted = ScriptedCompletion(by_id)
        self.name = name

    def __call__(self, spec: TaskSpec) -> Prediction:
        return parse_response(self._scripted.for_spec(spec))
