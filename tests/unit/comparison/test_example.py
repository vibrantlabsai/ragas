import json
import runpy
from pathlib import Path

import pytest

from ragas.comparison import compare
from ragas.comparison.io import read_jsonl, read_plan


@pytest.mark.asyncio
async def test_offline_example_and_exported_evidence_reproduce_decisions(tmp_path):
    example = Path(__file__).resolve().parents[3] / "examples" / "paired_comparison.py"
    namespace = runpy.run_path(str(example))
    reports = await namespace["run_example"](tmp_path)
    assert reports["improved"].verdict == "pass"
    assert reports["incomplete"].verdict == "inconclusive"
    plan = read_plan(tmp_path / "plan.json")
    replay = compare(
        read_jsonl(tmp_path / "baseline.jsonl"),
        read_jsonl(tmp_path / "incomplete.jsonl"),
        plan=plan,
    )
    assert replay.to_dict() == reports["incomplete"].to_dict()
    saved = json.loads((tmp_path / "incomplete.report.json").read_text())
    assert saved["comparisons"][0]["coverage"]["excluded_cases"] == 2
