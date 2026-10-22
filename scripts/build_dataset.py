"""Build data/benchmark.jsonl from the authored source, validating as we go."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leakbench._authored import ITEMS          # noqa: E402
from leakbench.config import DATASET_PATH       # noqa: E402
from leakbench.dataset import dump, integrity_report  # noqa: E402
from leakbench.schema import validate_item      # noqa: E402


def main() -> int:
    for item in ITEMS:
        validate_item(item)
    dump(ITEMS, DATASET_PATH)
    report = integrity_report(ITEMS)
    print(f"Wrote {len(ITEMS)} items to {DATASET_PATH}")
    print("Distribution:", report["distribution"])
    if not report["ok"]:
        print("INTEGRITY PROBLEMS:")
        for p in report["problems"]:
            print("  -", p)
        return 1
    print("Integrity: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
