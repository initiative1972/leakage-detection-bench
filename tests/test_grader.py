from leakbench.grader import grade
from leakbench.schema import BenchItem, Feature, Label, Prediction, TaskSpec


def _item(id, has, leak_type, offending):
    spec = TaskSpec(
        id=id, domain="d", task_type="classification", target="t",
        prediction_time="now", split="random", preprocessing="none",
        features=[Feature("a", "int", "x"), Feature("b", "int", "y")],
    )
    return BenchItem(spec, Label(has, leak_type, offending, "r", "easy"))


ITEMS = [
    _item("1", True, "target_leakage", ["a"]),
    _item("2", True, "temporal_leakage", ["b"]),
    _item("3", False, "none", []),
    _item("4", False, "none", []),
]


def test_perfect_predictions():
    preds = {
        "1": Prediction(True, "target_leakage", ["a"]),
        "2": Prediction(True, "temporal_leakage", ["b"]),
        "3": Prediction(False),
        "4": Prediction(False),
    }
    g = grade(ITEMS, preds).as_dict()
    assert g["detection"]["precision"] == 1.0
    assert g["detection"]["recall"] == 1.0
    assert g["detection"]["f1"] == 1.0
    assert g["diagnosis"]["type_accuracy"] == 1.0
    assert g["diagnosis"]["localisation_jaccard"] == 1.0
    assert g["aggregate"] == 1.0


def test_always_clean_scores_zero_aggregate():
    preds = {it.spec.id: Prediction(False) for it in ITEMS}
    g = grade(ITEMS, preds).as_dict()
    assert g["detection"]["recall"] == 0.0
    assert g["aggregate"] == 0.0


def test_detection_without_diagnosis_is_penalised():
    # correct detection, but wrong type and wrong feature
    preds = {
        "1": Prediction(True, "proxy_leakage", ["b"]),
        "2": Prediction(True, "proxy_leakage", ["a"]),
        "3": Prediction(False),
        "4": Prediction(False),
    }
    g = grade(ITEMS, preds).as_dict()
    assert g["detection"]["f1"] == 1.0
    assert g["diagnosis"]["type_accuracy"] == 0.0
    assert g["diagnosis"]["localisation_jaccard"] == 0.0
    # aggregate = 0.5*1 + 0.3*0 + 0.2*0
    assert abs(g["aggregate"] - 0.5) < 1e-9


def test_missing_prediction_counts_as_clean():
    preds = {"1": Prediction(True, "target_leakage", ["a"])}  # 2,3,4 missing
    g = grade(ITEMS, preds).as_dict()
    # item 2 leaky but missing -> false negative
    assert g["detection"]["fn"] == 1
    assert g["detection"]["tp"] == 1
    assert g["detection"]["tn"] == 2
