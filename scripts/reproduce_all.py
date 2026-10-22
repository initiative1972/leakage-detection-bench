"""Reproduce everything: (re)build the dataset, validate it, run the baselines,
render figures, and assert the headline findings so CI fails loudly if a change
breaks the story.

Fully deterministic and offline (pure standard library).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leakbench import figures                     # noqa: E402
from leakbench._authored import ITEMS             # noqa: E402
from leakbench.baselines import SOLVERS           # noqa: E402
from leakbench.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402
from leakbench.dataset import dump, integrity_report, load  # noqa: E402
from leakbench.evaluate import evaluate, leaderboard  # noqa: E402
from leakbench.schema import validate_item        # noqa: E402


def main() -> int:
    # 1) Build + validate dataset
    for item in ITEMS:
        validate_item(item)
    dump(ITEMS)
    items = load()
    report = integrity_report(items)

    # 2) Run baselines -> leaderboard
    results = [evaluate(items, solver, name) for name, solver in SOLVERS.items()]
    board = leaderboard(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "leaderboard.json").write_text(
        json.dumps(board, indent=2), encoding="utf-8"
    )

    # 3) Figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / "leaderboard.svg").write_text(
        figures.leaderboard_svg(board), encoding="utf-8"
    )
    by_type = {r["solver"]: r["recall_by_type"]
               for r in board if r["solver"] in ("keyword_naive", "heuristic")}
    (FIGURES_DIR / "recall_by_type.svg").write_text(
        figures.recall_by_type_svg(by_type), encoding="utf-8"
    )

    # 4) Gate the headline findings
    by = {r["solver"]: r for r in board}
    heur = by["heuristic"]
    naive = by["keyword_naive"]
    aleak = by["always_leak"]
    aclean = by["always_clean"]
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    dist = report["distribution"]
    check("dataset integrity (schema + canary + no near-dupes)", report["ok"])
    check("dataset is reasonably balanced (clean 30-55%)",
          0.30 <= dist["n_clean"] / dist["n_items"] <= 0.55)
    check("all five positive leak types are represented",
          len([t for t in dist["by_type"] if t != "none"]) == 5)
    check("curated linter beats naive keyword matcher (aggregate)",
          heur["aggregate"] > naive["aggregate"])
    check("naive over-flags: lower precision than the linter",
          naive["detection"]["precision"] < heur["detection"]["precision"])
    check("linter is strong but NOT perfect (recall < 1.0 -> headroom)",
          heur["detection"]["recall"] < 1.0)
    check("diagnosis is unsolved by rules (type_accuracy < 1.0 -> headroom)",
          heur["diagnosis"]["type_accuracy"] < 1.0)
    check("trivial 'always leak' is weak on aggregate (< 0.5)",
          aleak["aggregate"] < 0.5)
    check("baseline ordering: heuristic > naive > always_leak > always_clean",
          heur["aggregate"] > naive["aggregate"] > aleak["aggregate"]
          > aclean["aggregate"])

    print("\nDistribution:", json.dumps(dist))
    print("\nLeaderboard:")
    print(f"  {'solver':<16}{'F1':>8}{'prec':>8}{'rec':>8}"
          f"{'type':>8}{'loc':>8}{'agg':>8}")
    for r in board:
        d = r["detection"]
        print(f"  {r['solver']:<16}{d['f1']:>8.3f}{d['precision']:>8.3f}"
              f"{d['recall']:>8.3f}{r['diagnosis']['type_accuracy']:>8.3f}"
              f"{r['diagnosis']['localisation_jaccard']:>8.3f}"
              f"{r['aggregate']:>8.3f}")

    print("\nall_gates_passed:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
