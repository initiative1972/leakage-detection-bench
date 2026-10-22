"""Load and sanity-check the benchmark JSONL."""
from __future__ import annotations

import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

from .config import CANARY, DATASET_PATH
from .schema import BenchItem, item_from_dict, item_to_dict, validate_item


def load(path: Path = DATASET_PATH) -> List[BenchItem]:
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(item_from_dict(json.loads(line)))
    return items


def dump(items: List[BenchItem], path: Path = DATASET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps(item_to_dict(item), ensure_ascii=False))
            fh.write("\n")


def label_distribution(items: List[BenchItem]) -> dict:
    by_type = Counter(it.label.leak_type for it in items)
    by_diff = Counter(it.label.difficulty for it in items)
    n_leaky = sum(1 for it in items if it.label.has_leakage)
    return {
        "n_items": len(items),
        "n_leaky": n_leaky,
        "n_clean": len(items) - n_leaky,
        "by_type": dict(sorted(by_type.items())),
        "by_difficulty": dict(sorted(by_diff.items())),
    }


def near_duplicates(items: List[BenchItem], threshold: float = 0.92) -> List[tuple]:
    """Return (id_a, id_b, ratio) pairs whose public text is near-identical.

    A contamination control: two items that are near-duplicates would let a
    solver pattern-match rather than reason, and would double-count in scoring.
    """
    texts = [(it.spec.id, it.spec.public_text()) for it in items]
    dupes = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ratio = SequenceMatcher(None, texts[i][1], texts[j][1]).ratio()
            if ratio >= threshold:
                dupes.append((texts[i][0], texts[j][0], round(ratio, 3)))
    return dupes


def integrity_report(items: List[BenchItem]) -> dict:
    """Validate every item and run contamination checks; collect problems."""
    problems: List[str] = []

    ids = [it.spec.id for it in items]
    if len(ids) != len(set(ids)):
        problems.append("duplicate spec ids")

    for it in items:
        try:
            validate_item(it)
        except Exception as exc:  # noqa: BLE001 - surfaced as a problem string
            problems.append(str(exc))
        if it.canary != CANARY:
            problems.append(f"{it.spec.id}: missing/altered canary")

    dupes = near_duplicates(items)
    for a, b, r in dupes:
        problems.append(f"near-duplicate specs: {a} ~ {b} (ratio {r})")

    return {
        "ok": not problems,
        "problems": problems,
        "distribution": label_distribution(items),
        "near_duplicates": dupes,
    }
