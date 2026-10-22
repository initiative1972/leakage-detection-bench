import pytest

from leakbench._authored import ITEMS
from leakbench.schema import (
    BenchItem,
    Feature,
    Label,
    SchemaError,
    TaskSpec,
    item_from_dict,
    item_to_dict,
    validate_item,
)


def test_all_authored_items_validate():
    for item in ITEMS:
        validate_item(item)


def _spec(features):
    return TaskSpec(
        id="X1", domain="d", task_type="classification", target="t",
        prediction_time="now", split="random", preprocessing="none",
        features=features,
    )


def test_has_leakage_must_agree_with_type():
    spec = _spec([Feature("a", "int", "desc")])
    bad = BenchItem(spec, Label(True, "none", [], "r", "easy"))
    with pytest.raises(SchemaError):
        validate_item(bad)


def test_offending_feature_must_exist():
    spec = _spec([Feature("a", "int", "desc")])
    bad = BenchItem(spec, Label(True, "target_leakage", ["ghost"], "r", "easy"))
    with pytest.raises(SchemaError):
        validate_item(bad)


def test_clean_item_cannot_list_offending():
    spec = _spec([Feature("a", "int", "desc")])
    bad = BenchItem(spec, Label(False, "none", ["a"], "r", "easy"))
    with pytest.raises(SchemaError):
        validate_item(bad)


def test_roundtrip_serialization():
    for item in ITEMS[:5]:
        again = item_from_dict(item_to_dict(item))
        assert again.spec.id == item.spec.id
        assert again.label.leak_type == item.label.leak_type
        assert again.spec.feature_names() == item.spec.feature_names()
