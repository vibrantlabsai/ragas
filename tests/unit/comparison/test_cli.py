import json
import os
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from ragas.cli import app
from ragas.comparison import ComparisonPlan, MetricSpec, compare
from ragas.comparison.io import write_plan
from ragas.comparison.render import render_markdown


def inputs(tmp_path, before=1, after=1):
    baseline = tmp_path / "before.jsonl"
    candidate = tmp_path / "after.jsonl"
    plan = tmp_path / "plan.json"
    baseline.write_text(json.dumps({"case_id": "a", "score": before}) + "\n")
    candidate.write_text(json.dumps({"case_id": "a", "score": after}) + "\n")
    write_plan(plan, ComparisonPlan(metrics=(MetricSpec("score"),), mode="observed"))
    return baseline, candidate, plan


@pytest.mark.parametrize(
    "after,exit_code,verdict",
    [(1, 0, "pass"), (0, 1, "regress"), (None, 2, "inconclusive")],
)
def test_real_typer_command_and_machine_output(tmp_path, after, exit_code, verdict):
    baseline, candidate, plan = inputs(tmp_path, after=after)
    output = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--plan",
            str(plan),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == exit_code, result.output
    assert json.loads(output.read_text())["verdict"] == verdict


def test_invalid_input_does_not_echo_payload_or_exit_zero(tmp_path):
    baseline, candidate, plan = inputs(tmp_path)
    candidate.write_text('{"case_id":"a","score":"SECRET-PRIVATE-VALUE"}')
    result = CliRunner().invoke(
        app,
        [
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--plan",
            str(plan),
        ],
    )
    assert result.exit_code == 3
    assert "SECRET-PRIVATE-VALUE" not in result.output


def test_command_rejects_output_input_alias_before_writing(tmp_path):
    baseline, candidate, plan = inputs(tmp_path)
    original = baseline.read_bytes()
    result = CliRunner().invoke(
        app,
        [
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--plan",
            str(plan),
            "--output",
            str(baseline),
        ],
    )
    assert result.exit_code == 3
    assert baseline.read_bytes() == original


def test_markdown_escapes_case_ids_and_metric_names():
    name = "score|<script>"
    row = {"case_id": "[click](https://example.com)\n\x1b[31m", name: 1}
    report = compare(
        [row], [row], plan=ComparisonPlan(metrics=(MetricSpec(name),), mode="observed")
    )
    rendered = render_markdown(report)
    assert "<script>" not in rendered
    assert "\x1b" not in rendered
    assert "[click](https://example.com)" not in rendered
    assert "Supplied rows" in rendered


def test_subprocess_command_runs_without_model_calls(tmp_path):
    baseline, candidate, plan = inputs(tmp_path, after=0)
    command = [
        sys.executable,
        "-m",
        "ragas.comparison",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--plan",
        str(plan),
        "--format",
        "json",
    ]
    env = {
        key: value for key, value in os.environ.items() if not key.endswith("API_KEY")
    }
    result = subprocess.run(
        command, capture_output=True, text=True, env=env, timeout=30
    )
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["verdict"] == "regress"


def test_csv_to_junit_command_preserves_inconclusive_gate(tmp_path):
    import xml.etree.ElementTree as ET

    baseline, candidate, plan = inputs(tmp_path)
    baseline.write_text("case_id,score\na,1\n")
    candidate.write_text("case_id,score\na,\n")
    output = tmp_path / "report.xml"
    result = CliRunner().invoke(
        app,
        [
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--plan",
            str(plan),
            "--input-format",
            "csv",
            "--output",
            str(output),
            "--output-format",
            "junit",
        ],
    )
    assert result.exit_code == 2
    assert ET.parse(output).getroot().attrib["errors"] == "1"
