from __future__ import annotations

import subprocess
import sys
import typing as t
from dataclasses import dataclass, field

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics.base import MetricType, SingleTurnMetric


@dataclass
class StaticMetric(SingleTurnMetric):
    name: str = "static_score"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"user_input", "response"}}
    )

    def init(self, run_config):
        pass

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks):
        return 0.7


@dataclass
class LLMMetric(StaticMetric):
    name: str = "llm_score"

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks):
        llm_run = callbacks.on_llm_start(
            {"name": "fake-llm", "model_name": "fake-model"},
            ["score this sample"],
        )[0]
        llm_run.on_llm_end(
            LLMResult(
                generations=[[ChatGeneration(message=AIMessage(content="score: 0.9"))]],
                llm_output={
                    "token_usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 3,
                        "total_tokens": 14,
                    },
                    "model_name": "fake-model",
                },
            )
        )
        return 0.9


@dataclass
class FailingMetric(StaticMetric):
    name: str = "failing_score"

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks):
        raise ValueError("metric failed")


@pytest.fixture
def span_exporter():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    try:
        from opentelemetry.sdk.trace.export import InMemorySpanExporter
    except ImportError:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


def _dataset(rows: int = 3) -> EvaluationDataset:
    return EvaluationDataset(
        samples=[
            SingleTurnSample(user_input=f"question {i}", response=f"answer {i}")
            for i in range(rows)
        ]
    )


def _run_evaluation(metric: SingleTurnMetric, provider):
    from ragas.integrations.tracing.opentelemetry import OpenTelemetryTracer

    tracer = OpenTelemetryTracer(tracer=provider.get_tracer("ragas-test"))
    return evaluate(
        _dataset(),
        metrics=[metric],
        callbacks=[tracer],
        show_progress=False,
    )


def test_evaluate_emits_evaluation_row_and_metric_spans(span_exporter):
    exporter, provider = span_exporter
    _run_evaluation(StaticMetric(), provider)

    spans = exporter.get_finished_spans()
    evaluate_spans = [span for span in spans if span.name == "ragas.evaluate"]
    row_spans = [span for span in spans if span.name == "ragas.row"]
    metric_spans = [span for span in spans if span.name == "ragas.metric.static_score"]

    assert len(evaluate_spans) == 1
    assert len(row_spans) == 3
    assert len(metric_spans) == 3
    assert all(span.end_time > span.start_time for span in spans)
    assert {span.attributes["ragas.row.index"] for span in row_spans} == {0, 1, 2}
    assert all(span.attributes["ragas.metric.score"] == 0.7 for span in metric_spans)

    root_span_id = evaluate_spans[0].context.span_id
    assert all(span.parent.span_id == root_span_id for span in row_spans)


def test_llm_metric_emits_llm_span_with_token_usage(span_exporter):
    exporter, provider = span_exporter
    _run_evaluation(LLMMetric(), provider)

    llm_spans = [
        span for span in exporter.get_finished_spans() if span.name == "ragas.llm.call"
    ]

    assert llm_spans
    assert all(span.attributes["gen_ai.usage.input_tokens"] == 11 for span in llm_spans)
    assert all(span.attributes["gen_ai.usage.output_tokens"] == 3 for span in llm_spans)
    assert all(span.attributes["llm.model_name"] == "fake-model" for span in llm_spans)


def test_no_global_tracer_provider_does_not_break_evaluation():
    pytest.importorskip("opentelemetry.sdk")
    from ragas.integrations.tracing.opentelemetry import OpenTelemetryTracer

    tracer = OpenTelemetryTracer()
    result = evaluate(
        _dataset(rows=1),
        metrics=[StaticMetric()],
        callbacks=[tracer],
        show_progress=False,
    )

    assert result.scores == [{"static_score": 0.7}]


def test_missing_opentelemetry_dependencies_raise_install_hint():
    code = """
import sys
import types

vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
vertexai.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = vertexai

for name in list(sys.modules):
    if name == "opentelemetry" or name.startswith("opentelemetry."):
        del sys.modules[name]

class BlockOpenTelemetry:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "opentelemetry" or fullname.startswith("opentelemetry."):
            raise ImportError("blocked")
        return None

sys.meta_path.insert(0, BlockOpenTelemetry())

try:
    import ragas.integrations.tracing.opentelemetry
except ImportError as exc:
    assert "pip install ragas[opentelemetry]" in str(exc)
else:
    raise AssertionError("import should have failed")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_metric_failure_records_exception_and_error_status(span_exporter):
    from opentelemetry.trace import StatusCode

    exporter, provider = span_exporter
    result = _run_evaluation(FailingMetric(), provider)

    metric_spans = [
        span
        for span in exporter.get_finished_spans()
        if span.name == "ragas.metric.failing_score"
    ]

    assert len(metric_spans) == 3
    assert all(span.status.status_code == StatusCode.ERROR for span in metric_spans)
    assert all(
        any(event.name == "exception" for event in span.events) for span in metric_spans
    )
    assert len(result.scores) == 3
    assert all("failing_score" in score for score in result.scores)
