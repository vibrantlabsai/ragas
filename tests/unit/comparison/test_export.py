import json
import math

import pytest

from ragas.comparison import ComparisonLimits, ComparisonPlan, MetricSpec, compare
from ragas.comparison.export import write_observations
from ragas.comparison.io import read_jsonl


def test_export_selects_columns_and_replays_failed_status_without_payload(tmp_path):
    plan = ComparisonPlan(
        metrics=(MetricSpec("score"),), mode="observed", status_field="state"
    )
    rows = [
        {"case_id": "a", "score": 1, "state": "success", "secret": object()},
        {"case_id": "b", "score": 0, "state": "error", "prompt": "PRIVATE"},
    ]
    path = tmp_path / "rows.jsonl"
    assert write_observations(path, rows, plan=plan) == 2
    restored = list(read_jsonl(path))
    assert "PRIVATE" not in path.read_text()
    assert "secret" not in path.read_text()
    assert (
        compare(rows, rows, plan=plan).to_dict()
        == compare(restored, restored, plan=plan).to_dict()
    )


def test_export_normalizes_nan_to_null_and_preserves_missing_column(tmp_path):
    plan = ComparisonPlan(metrics=(MetricSpec("score"),), mode="observed")
    path = tmp_path / "rows.jsonl"
    write_observations(
        path, [{"case_id": "a", "score": math.nan}, {"case_id": "b"}], plan=plan
    )
    assert list(read_jsonl(path)) == [{"case_id": "a", "score": None}, {"case_id": "b"}]
    assert "NaN" not in path.read_text()


@pytest.mark.parametrize(
    "rows", [[{"case_id": "a", "score": True}], [{"case_id": "a"}, {"case_id": "a"}]]
)
def test_invalid_export_preserves_existing_file_and_leaves_no_temporary(tmp_path, rows):
    plan = ComparisonPlan(metrics=(MetricSpec("score"),), mode="observed")
    path = tmp_path / "rows.jsonl"
    path.write_text("existing")
    with pytest.raises(ValueError):
        write_observations(path, rows, plan=plan)
    assert path.read_text() == "existing"
    assert list(tmp_path.iterdir()) == [path]


def test_file_budget_failure_does_not_publish_partial_evidence(tmp_path):
    plan = ComparisonPlan(metrics=(MetricSpec("score"),), mode="observed")
    path = tmp_path / "rows.jsonl"
    with pytest.raises(ValueError, match="file byte"):
        write_observations(
            path,
            [{"case_id": "a", "score": 1}],
            plan=plan,
            limits=ComparisonLimits(max_file_bytes=10),
        )
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_alias_is_rejected_before_consuming_observations(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"case_id": "a", "score": 1}))
    plan = ComparisonPlan(metrics=(MetricSpec("score"),), mode="observed")
    with pytest.raises(ValueError, match="alias"):
        write_observations(path, read_jsonl(path), plan=plan, input_paths=(path,))
