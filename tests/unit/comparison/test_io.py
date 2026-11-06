import json

import pytest

from ragas.comparison import (
    ComparisonLimits,
    ComparisonPlan,
    ExpectedCase,
    MetricSpec,
    SliceSpec,
    compare,
)
from ragas.comparison.io import (
    plan_from_dict,
    plan_to_dict,
    read_jsonl,
    read_plan,
    write_plan,
    write_report,
)


def test_full_plan_roundtrip_preserves_typed_attributes(tmp_path):
    plan = ComparisonPlan(
        metrics=(MetricSpec("quality", "score", margin=0.1, minimum=0, maximum=1),),
        repeat_id="rep",
        case_fields=("tenant",),
        status_field="status",
        success_values=("ok",),
        slices=(SliceSpec("tenant", "tenant", (True, 1, 1.0)),),
        expected_cases=(ExpectedCase("a", ("r",), (("tenant", True),)),),
        mode="observed",
        minimum_coverage=0.9,
    )
    path = tmp_path / "plan.json"
    write_plan(path, plan)
    assert read_plan(path) == plan
    assert plan_from_dict(plan_to_dict(plan)) == plan


@pytest.mark.parametrize(
    "payload",
    [
        '{"format":"ragas.comparison.plan","format":"other","version":1,"plan":{}}',
        '{"format":"ragas.comparison.plan","version":99,"plan":{}}',
        '{"format":"ragas.comparison.plan","version":true,"plan":{}}',
        '{"format":"ragas.comparison.plan","version":1,"plan":{"metrics":[{"name":"x","oops":1}]}}',
        '{"format":"ragas.comparison.plan","version":1,"plan":{"metrics":[{"name":"x"}],"confidence":NaN}}',
    ],
)
def test_malformed_plan_cannot_silently_change_policy(tmp_path, payload):
    path = tmp_path / "plan.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        read_plan(path)


def test_plan_cannot_raise_trusted_resource_limits(tmp_path):
    path = tmp_path / "plan.json"
    write_plan(path, ComparisonPlan(metrics=(MetricSpec("a"), MetricSpec("b"))))
    with pytest.raises(ValueError, match="metrics"):
        read_plan(path, limits=ComparisonLimits(max_metrics=1))


def test_jsonl_streaming_limits_and_duplicate_keys(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"case_id":"a","score":1}\n\n{"case_id":"b","score":0}\n')
    assert len(list(read_jsonl(path))) == 2
    with pytest.raises(ValueError, match="rows"):
        list(read_jsonl(path, limits=ComparisonLimits(max_rows=1)))
    with pytest.raises(ValueError, match="line"):
        list(read_jsonl(path, limits=ComparisonLimits(max_line_bytes=8)))
    path.write_text('{"case_id":"a","score":1,"score":0}\n')
    with pytest.raises(ValueError, match="duplicate"):
        list(read_jsonl(path))


def test_json_structure_is_bounded(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"case_id": "a", "extra": [[[[[0]]]]]}))
    with pytest.raises(ValueError, match="depth"):
        list(read_jsonl(path, limits=ComparisonLimits(max_json_depth=3)))
    with pytest.raises(ValueError, match="nodes"):
        list(read_jsonl(path, limits=ComparisonLimits(max_json_nodes=3)))


def test_output_does_not_alias_inputs_or_leave_tempfiles(tmp_path, monkeypatch):
    import ragas.comparison.io as io

    path = tmp_path / "input.jsonl"
    path.write_text("original")
    report = compare(
        [{"case_id": "a", "x": 1}],
        [{"case_id": "a", "x": 1}],
        plan=ComparisonPlan(metrics=(MetricSpec("x"),), mode="observed"),
    )
    with pytest.raises(ValueError, match="alias"):
        write_report(path, report, input_paths=(path,))
    output = tmp_path / "report.json"
    output.write_text("previous report")

    def fail_replace(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(io.os, "replace", fail_replace)
    with pytest.raises(OSError):
        write_report(output, report)
    assert output.read_text() == "previous report"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["input.jsonl", "report.json"]


def test_report_json_records_counts_and_not_arbitrary_payload(tmp_path):
    report = compare(
        [{"case_id": "a", "x": 1, "secret": "never export"}],
        [{"case_id": "a", "x": 1}],
        plan=ComparisonPlan(metrics=(MetricSpec("x"),), mode="observed"),
    )
    path = tmp_path / "report.json"
    write_report(path, report)
    parsed = json.loads(path.read_text())
    assert parsed["verdict"] == "pass"
    assert "never export" not in path.read_text()
