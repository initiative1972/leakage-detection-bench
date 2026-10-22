"""Run every deterministic baseline and write the leaderboard JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leakbench.baselines import SOLVERS          # noqa: E402
from leakbench.config import RESULTS_DIR          # noqa: E402
from leakbench.dataset import load                # noqa: E402
from leakbench.evaluate import evaluate, leaderboard  # noqa: E402


def main() -> int:
    items = load()
    results = [evaluate(items, solver, name)
               for name, solver in SOLVERS.items()]
    board = leaderboard(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "leaderboard.json").write_text(
        json.dumps(board, indent=2), encoding="utf-8"
    )

    print(f"{'solver':<16}{'F1':>8}{'prec':>8}{'rec':>8}"
          f"{'type':>8}{'loc':>8}{'agg':>8}")
    for r in board:
        d = r["detection"]
        print(f"{r['solver']:<16}{d['f1']:>8.3f}{d['precision']:>8.3f}"
              f"{d['recall']:>8.3f}{r['diagnosis']['type_accuracy']:>8.3f}"
              f"{r['diagnosis']['localisation_jaccard']:>8.3f}"
              f"{r['aggregate']:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
