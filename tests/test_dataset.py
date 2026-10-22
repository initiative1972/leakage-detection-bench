from leakbench._authored import ITEMS
from leakbench.config import CANARY
from leakbench.dataset import integrity_report, load, near_duplicates


def test_loads_all_items():
    items = load()
    assert len(items) == len(ITEMS) == 36


def test_integrity_is_clean():
    report = integrity_report(load())
    assert report["ok"], report["problems"]


def test_no_near_duplicates():
    assert near_duplicates(load()) == []


def test_every_item_carries_the_canary():
    for it in load():
        assert it.canary == CANARY


def test_label_distribution_is_balanced_enough():
    dist = integrity_report(load())["distribution"]
    clean_frac = dist["n_clean"] / dist["n_items"]
    assert 0.30 <= clean_frac <= 0.55
    assert len([t for t in dist["by_type"] if t != "none"]) == 5
