"""Deterministic grader.

A solver is scored on three things, in increasing difficulty:

  1. **Detection** -- did it flag leaky specs as leaky? (precision / recall / F1
     over the binary has_leakage).
  2. **Diagnosis** -- for truly-leaky items, did it name the correct leak_type?
     Credit is given only if the item was also *detected*.
  3. **Localisation** -- for truly-leaky items whose type pins a feature
     (target / temporal / proxy), did it point at the right feature(s)?
     (Jaccard overlap; credit only if detected.)

The headline ``aggregate`` blends them so that merely shouting "leak!" on every
item cannot win: detection F1 is necessary but not sufficient -- you must also
diagnose and localise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .schema import BenchItem, Prediction

# Types for which a specific offending feature is expected.
LOCALISABLE = ("target_leakage", "temporal_leakage", "proxy_leakage")

# aggregate = w_det * F1 + w_diag * type_acc + w_loc * localisation
W_DETECTION = 0.5
W_DIAGNOSIS = 0.3
W_LOCALISATION = 0.2


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class GradeResult:
    n_items: int
    # detection
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int
    fp: int
    fn: int
    tn: int
    # diagnosis / localisation (over truly-leaky items)
    n_leaky: int
    type_accuracy: float      # fraction of leaky items detected AND typed right
    localisation: float       # mean Jaccard over localisable leaky items
    aggregate: float

    def as_dict(self) -> dict:
        return {
            "n_items": self.n_items,
            "detection": {
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1": round(self.f1, 4),
                "accuracy": round(self.accuracy, 4),
                "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            },
            "diagnosis": {
                "n_leaky": self.n_leaky,
                "type_accuracy": round(self.type_accuracy, 4),
                "localisation_jaccard": round(self.localisation, 4),
            },
            "aggregate": round(self.aggregate, 4),
        }


def grade(items: List[BenchItem], preds: Dict[str, Prediction]) -> GradeResult:
    """Grade a mapping ``spec_id -> Prediction`` against the labels."""
    tp = fp = fn = tn = 0
    n_leaky = 0
    type_hits = 0
    loc_scores: List[float] = []

    for it in items:
        pred = preds.get(it.spec.id, Prediction(has_leakage=False))
        true_leaky = it.label.has_leakage
        pred_leaky = pred.has_leakage

        if true_leaky and pred_leaky:
            tp += 1
        elif (not true_leaky) and pred_leaky:
            fp += 1
        elif true_leaky and (not pred_leaky):
            fn += 1
        else:
            tn += 1

        if true_leaky:
            n_leaky += 1
            detected = pred_leaky
            if detected and pred.leak_type == it.label.leak_type:
                type_hits += 1
            if it.label.leak_type in LOCALISABLE:
                if detected:
                    loc_scores.append(
                        _jaccard(pred.offending_features,
                                 it.label.offending_features)
                    )
                else:
                    loc_scores.append(0.0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = _f1(precision, recall)
    accuracy = (tp + tn) / len(items) if items else 0.0
    type_accuracy = type_hits / n_leaky if n_leaky else 0.0
    localisation = sum(loc_scores) / len(loc_scores) if loc_scores else 0.0

    aggregate = (
        W_DETECTION * f1
        + W_DIAGNOSIS * type_accuracy
        + W_LOCALISATION * localisation
    )

    return GradeResult(
        n_items=len(items), precision=precision, recall=recall, f1=f1,
        accuracy=accuracy, tp=tp, fp=fp, fn=fn, tn=tn, n_leaky=n_leaky,
        type_accuracy=type_accuracy, localisation=localisation,
        aggregate=aggregate,
    )


def grade_by_type(items: List[BenchItem], preds: Dict[str, Prediction]) -> dict:
    """Per-leak-type detection recall (how often each type is caught)."""
    from collections import defaultdict
    hit = defaultdict(int)
    total = defaultdict(int)
    for it in items:
        if not it.label.has_leakage:
            continue
        total[it.label.leak_type] += 1
        pred = preds.get(it.spec.id, Prediction(has_leakage=False))
        if pred.has_leakage:
            hit[it.label.leak_type] += 1
    return {t: round(hit[t] / total[t], 4) for t in sorted(total)}
