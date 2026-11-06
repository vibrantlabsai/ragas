import pytest
from pydantic import BaseModel

from ragas.comparison import ComparisonPlan, MetricSpec, compare
from ragas.experiment import Experiment


@pytest.mark.parametrize("typed", [False, True])
def test_real_experiment_delegate_is_pure_and_matches_direct_api(typed, monkeypatch):
    class Row(BaseModel):
        case_id: str
        score: float

    def row(score):
        return (
            Row(case_id="a", score=score) if typed else {"case_id": "a", "score": score}
        )

    before = Experiment("before", "inmemory", data=[row(0)])
    after = Experiment("after", "inmemory", data=[row(1)])
    monkeypatch.setattr(before, "save", lambda: pytest.fail("unexpected save"))
    monkeypatch.setattr(after, "save", lambda: pytest.fail("unexpected save"))
    plan = ComparisonPlan(metrics=(MetricSpec("score"),), mode="observed")
    assert after.compare(before, plan=plan) == compare(before, after, plan=plan)


def test_comparison_accepts_real_jsonl_backend_roundtrip(tmp_path):
    baseline = Experiment(
        "before", "local/jsonl", root_dir=tmp_path, data=[{"case_id": "a", "score": 0}]
    )
    candidate = Experiment(
        "after", "local/jsonl", root_dir=tmp_path, data=[{"case_id": "a", "score": 1}]
    )
    baseline.save()
    candidate.save()
    before = Experiment.load("before", "local/jsonl", root_dir=tmp_path)
    after = Experiment.load("after", "local/jsonl", root_dir=tmp_path)
    assert (
        after.compare(
            before, plan=ComparisonPlan(metrics=(MetricSpec("score"),), mode="observed")
        ).verdict
        == "pass"
    )
