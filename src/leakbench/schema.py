"""Typed schema for benchmark items, with validation and JSON (de)serialization.

A ``BenchItem`` has two halves:

  * ``spec`` -- everything a solver is allowed to see (the task description and
    its features); and
  * ``label`` -- the hidden answer key (is there a leak, which type, which
    features, and why).

Solvers receive only the ``spec`` (see ``spec.public_text()``); the grader
compares a solver's ``Prediction`` against the ``label``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List

from .config import CANARY, DIFFICULTIES, LEAK_TYPES, POSITIVE_TYPES


class SchemaError(ValueError):
    """Raised when an item violates the schema or label-consistency rules."""


@dataclass(frozen=True)
class Feature:
    name: str
    dtype: str
    description: str


@dataclass(frozen=True)
class TaskSpec:
    id: str
    domain: str
    task_type: str            # classification | regression | ranking
    target: str
    prediction_time: str      # when, in the real world, the prediction is made
    split: str                # how train/test are formed
    preprocessing: str        # how features are transformed
    features: List[Feature]
    notes: str = ""

    def public_text(self) -> str:
        """The full natural-language view a solver reasons over."""
        lines = [
            f"Domain: {self.domain}",
            f"Task type: {self.task_type}",
            f"Target: {self.target}",
            f"Prediction time: {self.prediction_time}",
            f"Split: {self.split}",
            f"Preprocessing: {self.preprocessing}",
            "Features:",
        ]
        for f in self.features:
            lines.append(f"  - {f.name} ({f.dtype}): {f.description}")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)

    def feature_names(self) -> List[str]:
        return [f.name for f in self.features]


@dataclass(frozen=True)
class Label:
    has_leakage: bool
    leak_type: str                       # one of LEAK_TYPES
    offending_features: List[str]
    rationale: str
    difficulty: str                      # easy | medium | hard


@dataclass(frozen=True)
class BenchItem:
    spec: TaskSpec
    label: Label
    canary: str = CANARY


@dataclass(frozen=True)
class Prediction:
    """A solver's answer for one item."""
    has_leakage: bool
    leak_type: str = "none"
    offending_features: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_item(item: BenchItem) -> None:
    """Raise ``SchemaError`` if an item is malformed or label-inconsistent."""
    s, lab = item.spec, item.label

    if not s.id:
        raise SchemaError("spec.id must be non-empty")
    if s.task_type not in ("classification", "regression", "ranking"):
        raise SchemaError(f"{s.id}: bad task_type '{s.task_type}'")
    if not s.features:
        raise SchemaError(f"{s.id}: must have at least one feature")
    if len(s.feature_names()) != len(set(s.feature_names())):
        raise SchemaError(f"{s.id}: duplicate feature names")

    if lab.leak_type not in LEAK_TYPES:
        raise SchemaError(f"{s.id}: unknown leak_type '{lab.leak_type}'")
    if lab.difficulty not in DIFFICULTIES:
        raise SchemaError(f"{s.id}: bad difficulty '{lab.difficulty}'")

    # has_leakage must agree with leak_type
    if lab.has_leakage and lab.leak_type == "none":
        raise SchemaError(f"{s.id}: has_leakage=True but leak_type='none'")
    if (not lab.has_leakage) and lab.leak_type != "none":
        raise SchemaError(f"{s.id}: has_leakage=False but leak_type="
                          f"'{lab.leak_type}'")

    # offending features must exist and be present only when leaky
    if lab.has_leakage:
        if lab.leak_type in ("target_leakage", "temporal_leakage",
                             "proxy_leakage") and not lab.offending_features:
            raise SchemaError(f"{s.id}: {lab.leak_type} requires "
                              f"offending_features")
        for fn in lab.offending_features:
            if fn not in s.feature_names():
                raise SchemaError(f"{s.id}: offending feature '{fn}' not in "
                                  f"feature list")
    else:
        if lab.offending_features:
            raise SchemaError(f"{s.id}: clean item must not list "
                              f"offending_features")
    if not lab.rationale.strip():
        raise SchemaError(f"{s.id}: rationale must be non-empty")


# --------------------------------------------------------------------------- #
# (De)serialization
# --------------------------------------------------------------------------- #
def item_to_dict(item: BenchItem) -> dict:
    d = asdict(item)
    return d


def item_from_dict(d: dict) -> BenchItem:
    spec_d = d["spec"]
    features = [Feature(**f) for f in spec_d["features"]]
    spec = TaskSpec(
        id=spec_d["id"],
        domain=spec_d["domain"],
        task_type=spec_d["task_type"],
        target=spec_d["target"],
        prediction_time=spec_d["prediction_time"],
        split=spec_d["split"],
        preprocessing=spec_d["preprocessing"],
        features=features,
        notes=spec_d.get("notes", ""),
    )
    lab_d = d["label"]
    label = Label(
        has_leakage=lab_d["has_leakage"],
        leak_type=lab_d["leak_type"],
        offending_features=list(lab_d["offending_features"]),
        rationale=lab_d["rationale"],
        difficulty=lab_d["difficulty"],
    )
    return BenchItem(spec=spec, label=label, canary=d.get("canary", CANARY))


def prediction_from_dict(d: dict) -> Prediction:
    return Prediction(
        has_leakage=bool(d["has_leakage"]),
        leak_type=d.get("leak_type", "none"),
        offending_features=list(d.get("offending_features", [])),
    )


__all__ = [
    "Feature", "TaskSpec", "Label", "BenchItem", "Prediction", "SchemaError",
    "validate_item", "item_to_dict", "item_from_dict", "prediction_from_dict",
    "LEAK_TYPES", "POSITIVE_TYPES",
]
