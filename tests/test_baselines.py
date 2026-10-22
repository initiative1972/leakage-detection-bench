from leakbench.baselines import SOLVERS, always_clean, heuristic, keyword_naive
from leakbench.dataset import load
from leakbench.evaluate import evaluate

ITEMS = load()


def _res(name):
    return evaluate(ITEMS, SOLVERS[name], name)


def test_always_clean_flags_nothing():
    for it in ITEMS:
        assert always_clean(it.spec).has_leakage is False


def test_heuristic_is_precise_but_imperfect_recall():
    r = _res("heuristic")
    assert r["detection"]["precision"] == 1.0
    assert r["detection"]["recall"] < 1.0        # leaves a subtle tail
    assert r["detection"]["f1"] > 0.9


def test_keyword_naive_overflags():
    r = _res("keyword_naive")
    # recall is high, but precision is lower than the curated linter
    assert r["detection"]["recall"] == 1.0
    assert r["detection"]["precision"] < 1.0


def test_naive_misdiagnoses_group_as_proxy():
    # a known shallow-pattern failure: id columns blamed, group leak missed
    group_items = [it for it in ITEMS if it.label.leak_type == "group_leakage"]
    assert group_items
    for it in group_items:
        pred = keyword_naive(it.spec)
        assert pred.has_leakage is True
        assert pred.leak_type != "group_leakage"


def test_heuristic_localises_a_target_leak():
    tl01 = next(it for it in ITEMS if it.spec.id == "TL01")
    pred = heuristic(tl01.spec)
    assert pred.leak_type == "target_leakage"
    assert set(pred.offending_features) & set(tl01.label.offending_features)


def test_leaderboard_ordering():
    agg = {n: _res(n)["aggregate"] for n in SOLVERS}
    assert agg["heuristic"] > agg["keyword_naive"] > agg["always_leak"] \
        > agg["always_clean"]
