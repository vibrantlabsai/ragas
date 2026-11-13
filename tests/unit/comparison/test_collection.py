import asyncio

import pytest
from pydantic import BaseModel

from ragas.comparison import ComparisonPlan, MetricSpec, SliceSpec, compare
from ragas.comparison.collection import ComparisonAdapter, plan_for_dataset
from ragas.executor import Executor
from ragas.run_config import RunConfig


def plan(**kwargs):
    return ComparisonPlan(metrics=(MetricSpec("score"),), mode="observed", **kwargs)


def test_manifest_is_prebuilt_without_scoring_and_is_detached():
    rows = [
        {"case_id": "a", "lang": "en", "prompt": object()},
        {"case_id": "b", "lang": "de"},
    ]
    p = plan_for_dataset(rows, plan(slices=(SliceSpec("en", "lang", ("en",)),)))
    rows[0]["lang"] = "fr"
    assert p.expected_cases[0].attributes == (("lang", "en"),)
    assert p.status_field == "__ragas_comparison_status"
    assert len(p.expected_cases) == 2


def test_manifest_repeated_cases_require_explicit_repeat_ids():
    with pytest.raises(ValueError, match="repeat"):
        plan_for_dataset([{"case_id": "a"}], plan(repeat_id="rep"))
    p = plan_for_dataset(
        [{"case_id": "a"}], plan(repeat_id="rep"), repeat_ids=("1", "2")
    )
    assert p.expected_cases[0].repeats == ("1", "2")


@pytest.mark.parametrize(
    "rows", [[], [{"case_id": "a"}, {"case_id": "a"}], [{"case_id": 1}]]
)
def test_all_manifest_validation_happens_before_any_adapter(rows):
    with pytest.raises(ValueError):
        plan_for_dataset(rows, plan())


@pytest.mark.asyncio
async def test_native_executor_retains_failed_results_and_missing_jobs():
    dataset = [{"case_id": "a"}, {"case_id": "b"}, {"case_id": "c"}]
    p = plan_for_dataset(dataset, plan())

    async def baseline(row):
        return {"score": 0.5}

    async def candidate(row):
        if row["case_id"] == "b":
            raise RuntimeError("PRIVATE EXCEPTION TEXT")
        return {"score": 0.7}

    before_adapter = ComparisonAdapter(baseline, plan=p)
    after_adapter = ComparisonAdapter(candidate, plan=p)
    before = Executor(show_progress=False, run_config=RunConfig(max_workers=2))
    after = Executor(show_progress=False, run_config=RunConfig(max_workers=2))
    for row in dataset:
        before.submit(before_adapter, row)
    for row in dataset[:2]:
        after.submit(after_adapter, row)
    b, c = await before.aresults(), await after.aresults()
    assert len(c) == 2
    assert c[1][p.status_field] == "error"
    assert "PRIVATE EXCEPTION TEXT" not in repr(c)
    report = compare(b, c, plan=p)
    assert report.verdict == "inconclusive"
    assert report.comparisons[0].coverage.total_cases == 3
    assert report.comparisons[0].coverage.eligible_cases == 1


@pytest.mark.asyncio
async def test_timeout_is_retained_but_external_cancellation_propagates():
    p = plan_for_dataset([{"case_id": "a"}], plan())
    stopped = asyncio.Event()

    async def hanging(row):
        try:
            await asyncio.sleep(10)
        finally:
            stopped.set()

    adapter = ComparisonAdapter(hanging, plan=p, timeout=0.001)
    result = await adapter({"case_id": "a"})
    assert result[p.status_field] == "timeout"
    assert stopped.is_set()
    task = asyncio.create_task(ComparisonAdapter(hanging, plan=p)({"case_id": "a"}))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output", [{"case_id": "other", "score": 1}, {"score": "1"}, None]
)
async def test_invalid_output_cannot_forge_identity_or_success(output):
    p = plan_for_dataset([{"case_id": "a"}], plan())

    async def evaluator(row):
        return output

    result = await ComparisonAdapter(evaluator, plan=p)({"case_id": "a"})
    assert result["case_id"] == "a"
    assert result[p.status_field] == "invalid_output"


@pytest.mark.asyncio
async def test_repeat_id_forwarding_and_model_metric_results():
    from ragas.metrics.result import MetricResult

    class Result(BaseModel):
        score: MetricResult

    p = plan_for_dataset([{"case_id": "a"}], plan(repeat_id="rep"), repeat_ids=("r",))

    async def evaluator(row, *, repeat_id):
        assert repeat_id == "r"
        return Result(score=MetricResult(0.8, reason="not exported"))

    result = await ComparisonAdapter(evaluator, plan=p, pass_repeat_id=True)(
        {"case_id": "a"}, repeat_id="r"
    )
    assert result["score"] == 0.8
    assert result[p.status_field] == "success"
    assert "not exported" not in repr(result)


@pytest.mark.asyncio
async def test_unknown_case_and_changed_attributes_reject_before_invocation():
    called = False

    async def evaluator(row):
        nonlocal called
        called = True
        return {"score": 1}

    p = plan_for_dataset([{"case_id": "a", "lang": "en"}], plan(case_fields=("lang",)))
    adapter = ComparisonAdapter(evaluator, plan=p)
    with pytest.raises(ValueError):
        await adapter({"case_id": "a", "lang": "de"})
    assert not called


@pytest.mark.asyncio
async def test_adapter_does_not_rescan_manifest_for_every_completed_job(monkeypatch):
    p = plan_for_dataset([{"case_id": str(i)} for i in range(1000)], plan())
    validations = 0
    original = ComparisonPlan.check_limits

    def counted(self, limits):
        nonlocal validations
        validations += 1
        return original(self, limits)

    monkeypatch.setattr(ComparisonPlan, "check_limits", counted)

    async def evaluator(row):
        return {"score": 1}

    adapter = ComparisonAdapter(evaluator, plan=p)
    for index in range(20):
        result = await adapter({"case_id": str(index)})
        assert result[p.status_field] == "success"
    assert validations == 1
