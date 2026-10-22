"""Deterministic, dependency-free baselines (no network, no models).

Four reference solvers, from trivial to a curated rule linter:

  * ``always_clean``   -- predicts "no leak" everywhere (the base-rate floor).
  * ``always_leak``    -- predicts "leak" everywhere (shows F1 is gameable, but
                          diagnosis/localisation are not).
  * ``keyword_naive``  -- flags on a broad list of suspicious tokens and treats
                          any id-like column as a proxy; high recall but it
                          over-flags benign cue words (precision suffers) and
                          mis-blames ids.
  * ``heuristic``      -- a curated leakage linter with negation handling and
                          structural rules; precise, but it still misses the
                          subtle tail (group leakage with no explicit
                          multiplicity cue, ordering/positional proxies).

The gap each leaves -- false positives for the naive detector, and a
subtle-leak tail plus imperfect diagnosis for the linter -- is exactly the
headroom a reasoning model (see ``llm_adapter``) has to earn.

A solver is any callable ``TaskSpec -> Prediction``.
"""
from __future__ import annotations

from typing import List

from .schema import Prediction, TaskSpec


def _text(spec: TaskSpec) -> str:
    return spec.public_text().lower()


def _is_id_like(name: str) -> bool:
    n = name.lower()
    return n.endswith("_id") or "index" in n or n in ("mrn", "record_id")


# --------------------------------------------------------------------------- #
# Trivial baselines
# --------------------------------------------------------------------------- #
def always_clean(spec: TaskSpec) -> Prediction:
    return Prediction(has_leakage=False)


def always_leak(spec: TaskSpec) -> Prediction:
    # A fixed, uninformed guess: shout "leak" and guess the single most common
    # type. Detection recall is perfect; diagnosis/localisation are not.
    return Prediction(has_leakage=True, leak_type="target_leakage",
                      offending_features=[])


# --------------------------------------------------------------------------- #
# keyword_naive -- broad, trippable
# --------------------------------------------------------------------------- #
_NAIVE_CUES = {
    "target_leakage": ["charged off", "charge off", "recover", "collections",
                       "chargeback", "readmission", "cancellation", "cancel",
                       "dispute", "refund"],
    "temporal_leakage": ["after", "future", "during", "months after"],
    "preprocessing_leakage": ["all rows", "entire dataset", "full dataset",
                              "over all rows", "train and test together",
                              "before the train", "before folds"],
    "proxy_leakage": ["bulk-loaded", "contiguous", "unshuffled", "shuffling",
                      "vendor", "source file", "sorted"],
}
# Priority order when several categories match.
_PRIORITY = ("target_leakage", "temporal_leakage", "preprocessing_leakage",
             "proxy_leakage", "group_leakage")


def keyword_naive(spec: TaskSpec) -> Prediction:
    text = _text(spec)
    matched = {}
    for leak_type, cues in _NAIVE_CUES.items():
        hits = [c for c in cues if c in text]
        if hits:
            matched[leak_type] = hits

    # Naive heuristic: any id-like column is treated as a proxy risk.
    id_features = [f.name for f in spec.features if _is_id_like(f.name)]
    if id_features:
        matched.setdefault("proxy_leakage", []).append("id-column")

    # Naive heuristic: any random split is treated as group risk.
    if "random" in spec.split.lower():
        matched.setdefault("group_leakage", [])

    if not matched:
        return Prediction(has_leakage=False)

    leak_type = next(t for t in _PRIORITY if t in matched)

    offending: List[str] = []
    if leak_type in ("target_leakage", "temporal_leakage"):
        cues = _NAIVE_CUES[leak_type]
        for f in spec.features:
            if any(c in f.description.lower() for c in cues):
                offending.append(f.name)
    elif leak_type == "proxy_leakage":
        cues = _NAIVE_CUES["proxy_leakage"]
        for f in spec.features:
            if _is_id_like(f.name) or any(c in f.description.lower()
                                          for c in cues):
                offending.append(f.name)

    return Prediction(has_leakage=True, leak_type=leak_type,
                      offending_features=offending)


# --------------------------------------------------------------------------- #
# heuristic -- curated linter with negation + structure
# --------------------------------------------------------------------------- #
_TARGET_CUES = ["charged off", "charge off", "recovered by collections",
                "recovery", "collections call", "chargeback",
                "readmission encounter", "readmission", "cancellation",
                "calls to cancel", "disput", "refund"]
_TEMPORAL_CUES = ["after the forecast", "months after", "during the entire",
                  "during this session", "finalised at month end",
                  "snapshot taken", "accrues"]
_PREP_LEAK_CUES = ["entire dataset before", "before the train/test split",
                   "over all rows", "train and test together",
                   "full dataset", "before folds are created",
                   "computed on all rows", "all rows (train and test"]
_PREP_SAFE_CUES = ["within each training fold", "on train only", "within folds",
                   "per training fold", "per fold", "fit on the training split",
                   "fit on train"]
_GROUP_MULTIPLICITY = ["multiple", "several", "per patient", "per machine",
                       "per user", "rows per", "re-apply", "contributes",
                       "many images", "studies over time"]
_PROXY_CUES = ["auto-increment", "bulk-loaded", "load order", "contiguous",
               "block-allocated", "assigned a contiguous", "supplied only",
               "vendor file", "which vendor"]


def _first_feature_matching(spec: TaskSpec, cues) -> List[str]:
    out = []
    for f in spec.features:
        d = f.description.lower()
        if any(c in d for c in cues):
            out.append(f.name)
    return out


def heuristic(spec: TaskSpec) -> Prediction:
    text = _text(spec)
    prep = spec.preprocessing.lower()
    split = spec.split.lower()

    # 1) target leakage -- a feature describes a post-outcome consequence
    tgt = _first_feature_matching(spec, _TARGET_CUES)
    if tgt:
        return Prediction(True, "target_leakage", tgt)

    # 2) temporal leakage -- a feature uses post-prediction information
    tmp = _first_feature_matching(spec, _TEMPORAL_CUES)
    if tmp:
        return Prediction(True, "temporal_leakage", tmp)

    # 3) proxy leakage -- an id/ordering/source encodes the label
    pxy = _first_feature_matching(spec, _PROXY_CUES)
    if pxy:
        return Prediction(True, "proxy_leakage", pxy)

    # 4) preprocessing leakage -- global fit before split, unless scoped safely
    if any(c in prep for c in _PREP_LEAK_CUES) and not any(
            c in prep for c in _PREP_SAFE_CUES):
        return Prediction(True, "preprocessing_leakage", [])

    # 5) group leakage -- random/row-level split AND explicit entity repetition
    row_level = ("random" in split) or ("individual" in split) or (
        "over" in split and "id" not in split)
    if row_level and any(c in text for c in _GROUP_MULTIPLICITY):
        return Prediction(True, "group_leakage", [])

    return Prediction(has_leakage=False)


SOLVERS = {
    "always_clean": always_clean,
    "always_leak": always_leak,
    "keyword_naive": keyword_naive,
    "heuristic": heuristic,
}
