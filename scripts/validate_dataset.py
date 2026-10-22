"""Validate data/benchmark.jsonl: schema, label consistency, canary, duplicates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leakbench.dataset import integrity_report, load  # noqa: E402


def main() -> int:
    items = load()
    report = integrity_report(items)
    print(json.dumps(report["distribution"], indent=2))
    if report["near_duplicates"]:
        print("Near-duplicates:", report["near_duplicates"])
    if not report["ok"]:
        print("INTEGRITY PROBLEMS:")
        for p in report["problems"]:
            print("  -", p)
        return 1
    print("Integrity: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
