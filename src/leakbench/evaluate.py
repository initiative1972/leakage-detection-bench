"""Run a solver over the benchmark and assemble leaderboard rows."""
from __future__ import annotations

from typing import Callable, Dict, List

from .grader import grade, grade_by_type
from .schema import BenchItem, Prediction, TaskSpec

Solver = Callable[[TaskSpec], Prediction]


def run_solver(items: List[BenchItem], solver: Solver) -> Dict[str, Prediction]:
    return {it.spec.id: solver(it.spec) for it in items}


def evaluate(items: List[BenchItem], solver: Solver, name: str) -> dict:
    preds = run_solver(items, solver)
    result = grade(items, preds).as_dict()
    result["solver"] = name
    result["recall_by_type"] = grade_by_type(items, preds)
    return result


def leaderboard(results: List[dict]) -> List[dict]:
    """Sort solver results by aggregate (desc), then F1."""
    rows = sorted(
        results,
        key=lambda r: (r["aggregate"], r["detection"]["f1"]),
        reverse=True,
    )
    return rows
