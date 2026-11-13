"""Offline paired evaluation with native Executor and retained failures.

Run: python examples/paired_comparison.py --output-dir /tmp/ragas-comparison
No model or external service is called.
"""

import argparse
import asyncio
from pathlib import Path

from ragas import Dataset
from ragas.comparison import (
    ComparisonAdapter,
    ComparisonPlan,
    MetricSpec,
    compare,
    plan_for_dataset,
    write_observations,
)
from ragas.comparison.io import write_plan, write_report
from ragas.executor import Executor
from ragas.run_config import RunConfig


async def baseline(row, *, repeat_id):
    index = row["index"]
    return {"quality": 0.5 + (index % 3) / 10, "cost": 10 + index % 4}


async def candidate(row, *, repeat_id):
    original = await baseline(row, repeat_id=repeat_id)
    return {
        "quality": original["quality"] + 0.08 + (row["index"] % 3) / 100,
        "cost": original["cost"] - 0.5 - (row["index"] % 2) / 10,
    }


async def faulty_candidate(row, *, repeat_id):
    if row["index"] == 28:
        raise RuntimeError("synthetic failure; error text is not exported")
    return await candidate(row, repeat_id=repeat_id)


async def collect(dataset, evaluator, plan, *, omit_last=False):
    adapter = ComparisonAdapter(evaluator, plan=plan, pass_repeat_id=True)
    executor = Executor(
        show_progress=False,
        raise_exceptions=True,
        run_config=RunConfig(max_workers=4),
        batch_size=20,
    )
    for row in dataset:
        if omit_last and row["index"] == 29:
            continue
        for repeat_id in ("first", "second"):
            executor.submit(adapter, row, repeat_id=repeat_id)
    return await executor.aresults()


async def run_example(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(
        "paired-offline-example",
        "inmemory",
        data=[{"case_id": f"case-{index:02d}", "index": index} for index in range(30)],
    )
    plan = plan_for_dataset(
        dataset,
        ComparisonPlan(
            metrics=(
                MetricSpec("quality", minimum=0, maximum=1),
                MetricSpec("cost", direction="lower", minimum=0, maximum=100),
            ),
            repeat_id="repeat",
            case_fields=("index",),
            seed=17,
        ),
        repeat_ids=("first", "second"),
    )
    before = await collect(dataset, baseline, plan)
    improved = await collect(dataset, candidate, plan)
    incomplete = await collect(dataset, faulty_candidate, plan, omit_last=True)
    write_plan(output_dir / "plan.json", plan)
    for name, rows in (
        ("baseline", before),
        ("improved", improved),
        ("incomplete", incomplete),
    ):
        write_observations(output_dir / f"{name}.jsonl", rows, plan=plan)
    reports = {}
    for name, rows in (("improved", improved), ("incomplete", incomplete)):
        report = compare(before, rows, plan=plan)
        reports[name] = report
        for format_name, extension in (
            ("json", "json"),
            ("markdown", "md"),
            ("junit", "xml"),
            ("csv", "csv"),
        ):
            write_report(
                output_dir / f"{name}.report.{extension}",
                report,
                output_format=format_name,
            )
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    reports = asyncio.run(run_example(arguments.output_dir))
    for name, report in reports.items():
        print(f"{name}: {report.verdict}")
