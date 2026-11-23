"""Run the Langfuse cookbook's tracing cells with the real SDK and offline scores."""

import re
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_langfuse_v4_cookbook_records_scores_on_the_rag_trace(monkeypatch):
    langfuse = pytest.importorskip("langfuse", minversion="4.7.0")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    demo_key = "pk-lf-ragas-cookbook-test"
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", demo_key)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-ragas-cookbook-test")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    exporter = InMemorySpanExporter()
    client = langfuse.Langfuse(
        base_url="http://127.0.0.1:1",
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
    )
    recorded = []

    def record_score(self, *, trace_id, name, value, **kwargs):
        recorded.append((trace_id, name, value))

    monkeypatch.setattr(langfuse.Langfuse, "create_score", record_score)

    async def score_with_ragas(question, contexts, answer):
        assert question == "What is the synthetic city?"
        assert contexts == ["The synthetic city is Bergen."]
        assert answer == "Bergen"
        return {"faithfulness": 1.0, "answer_relevancy": 0.75}

    question = "What is the synthetic city?"
    context = {
        "hash_string": lambda text: text,
        "q_to_c": {question: ["The synthetic city is Bergen."]},
        "q_to_a": {question: "Bergen"},
        "score_with_ragas": score_with_ragas,
    }
    cookbook = (
        Path(__file__).resolve().parents[3] / "docs/howtos/integrations/_langfuse.md"
    )
    cells = re.findall(
        r"```python\n(.*?)```", cookbook.read_text(encoding="utf-8"), re.DOTALL
    )
    setup = next(cell for cell in cells if "langfuse =" in cell)
    pipeline = next(cell for cell in cells if "def rag_pipeline(" in cell)

    try:
        exec(compile(setup, str(cookbook), "exec"), context)
        exec(compile(pipeline, str(cookbook), "exec"), context)
        assert await context["rag_pipeline"](question) == "Bergen"
        client.flush()

        spans = {span.name: span for span in exporter.get_finished_spans()}
        root = spans["rag_pipeline"]
        assert spans["retrieve"].parent.span_id == root.context.span_id
        assert spans["generator"].parent.span_id == root.context.span_id
        trace_id = f"{root.context.trace_id:032x}"
        assert recorded == [
            (trace_id, "faithfulness", 1.0),
            (trace_id, "answer_relevancy", 0.75),
        ]
    finally:
        client.shutdown()
