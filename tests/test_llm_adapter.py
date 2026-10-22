from leakbench.dataset import load
from leakbench.evaluate import evaluate
from leakbench.llm_adapter import ScriptedJudge, build_prompt, parse_response
from leakbench.schema import TaskSpec


def _spec():
    return TaskSpec(
        id="X", domain="d", task_type="classification", target="t",
        prediction_time="now", split="random", preprocessing="none",
        features=[],
    )


def test_prompt_contains_taxonomy_and_spec():
    p = build_prompt(_spec())
    assert "target_leakage" in p and "group_leakage" in p
    assert "Target: t" in p


def test_parse_tolerates_surrounding_prose():
    raw = ('Sure! Here is my answer:\n'
           '{"has_leakage": true, "leak_type": "temporal_leakage", '
           '"offending_features": ["f1"], "rationale": "future info"}\n'
           'Hope that helps.')
    pred = parse_response(raw)
    assert pred.has_leakage is True
    assert pred.leak_type == "temporal_leakage"
    assert pred.offending_features == ["f1"]


def test_parse_coerces_unknown_type():
    pred = parse_response('{"has_leakage": true, "leak_type": "weird"}')
    assert pred.has_leakage is True
    assert pred.leak_type == "target_leakage"


def test_scripted_judge_runs_end_to_end():
    items = load()
    # Give a perfect scripted answer for TL01; everything else defaults clean.
    tl01 = next(it for it in items if it.spec.id == "TL01")
    canned = {
        "TL01": '{"has_leakage": true, "leak_type": "target_leakage", '
                '"offending_features": '
                f'{list(tl01.label.offending_features)!r}'.replace("'", '"')
                + ', "rationale": "post-outcome"}'
    }
    judge = ScriptedJudge(canned, name="scripted")
    res = evaluate(items, judge, "scripted")
    # It detects exactly one leak (TL01) correctly.
    assert res["detection"]["tp"] == 1
    assert res["detection"]["fp"] == 0
