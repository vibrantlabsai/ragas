"""OpenTelemetry tracing for Ragas evaluations."""

from __future__ import annotations

__all__ = ["OpenTelemetryTracer", "RagasTracer"]

import math
import typing as t
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from ragas.callbacks import ChainType

INSTALL_HINT = (
    "Install OpenTelemetry tracing support with `pip install ragas[opentelemetry]`."
)

try:
    import opentelemetry.sdk.trace as _otel_sdk_trace  # noqa: F401
    from opentelemetry import context as context_api, trace
    from opentelemetry.trace import Status, StatusCode
except ImportError as exc:
    raise ImportError(INSTALL_HINT) from exc


Span = t.Any
Context = t.Any
Tracer = t.Any


def _chain_type_value(metadata: t.Optional[t.Dict[str, t.Any]]) -> t.Optional[str]:
    if not metadata:
        return None

    chain_type = metadata.get("type")
    if isinstance(chain_type, ChainType):
        return chain_type.value
    if isinstance(chain_type, str):
        return chain_type
    return None


def _safe_attribute(value: t.Any) -> t.Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (str, bool, int, float)) for item in value
    ):
        return list(value)
    return str(value)


def _get_nested(mapping: t.Mapping[str, t.Any], paths: t.Iterable[str]) -> t.Any:
    for path in paths:
        current: t.Any = mapping
        for key in path.split("."):
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return current
    return None


def _generation_metadata(response: LLMResult) -> t.Dict[str, t.Any]:
    if not response.generations or not response.generations[0]:
        return {}

    generation = response.generations[0][0]
    message = getattr(generation, "message", None)
    if message is not None:
        metadata = getattr(message, "response_metadata", None)
        if isinstance(metadata, dict):
            return metadata

    generation_info = getattr(generation, "generation_info", None)
    if isinstance(generation_info, dict):
        return generation_info

    return {}


@dataclass
class OpenTelemetryTracer(BaseCallbackHandler):
    """LangChain callback handler that emits OpenTelemetry spans for Ragas runs."""

    tracer_name: str = "ragas"
    tracer: t.Optional[Tracer] = None
    _spans: t.Dict[str, Span] = field(default_factory=dict, init=False, repr=False)
    _contexts: t.Dict[str, Context] = field(
        default_factory=dict, init=False, repr=False
    )
    _parents: t.Dict[str, t.Optional[str]] = field(
        default_factory=dict, init=False, repr=False
    )
    _row_indexes: t.Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _chain_types: t.Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tracer is None:
            self.tracer = trace.get_tracer(self.tracer_name)

    def shutdown(self) -> None:
        """End any spans that never received a terminal callback.

        Long-running evaluation runs can leave spans alive when a chain or
        LLM call ends without firing on_*_end (e.g. cancelled run, callback
        registration race). Without this method, the SDK keeps them in memory
        until process exit. Call shutdown() between evaluations or on tracer
        teardown to drain.
        """
        for run_id, span in list(self._spans.items()):
            try:
                span.set_status(Status(StatusCode.UNSET, "ended at shutdown"))
                span.end()
            except Exception:
                # End-of-life best-effort; do not raise during teardown.
                pass
            self._spans.pop(run_id, None)
        self._contexts.clear()
        self._parents.clear()
        self._row_indexes.clear()
        self._chain_types.clear()

    def on_chain_start(
        self,
        serialized: t.Dict[str, t.Any],
        inputs: t.Dict[str, t.Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: t.Optional[uuid.UUID] = None,
        tags: t.Optional[t.List[str]] = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        **kwargs: t.Any,
    ) -> t.Any:
        run_id_str = str(run_id)
        parent_id = str(parent_run_id) if parent_run_id else None
        self._parents[run_id_str] = parent_id
        chain_type = _chain_type_value(metadata)
        if chain_type is not None:
            self._chain_types[run_id_str] = chain_type

        span_name = self._span_name(serialized.get("name", ""), metadata)
        if span_name is None:
            return

        span = self.tracer.start_span(  # type: ignore[union-attr]
            span_name,
            context=self._parent_context(parent_id),
        )
        self._spans[run_id_str] = span
        self._contexts[run_id_str] = trace.set_span_in_context(span)

        self._set_common_attributes(
            span, serialized, inputs, run_id_str, parent_id, metadata
        )

    def on_chain_end(
        self,
        outputs: t.Dict[str, t.Any],
        *,
        run_id: uuid.UUID,
        **kwargs: t.Any,
    ) -> t.Any:
        run_id_str = str(run_id)
        span = self._spans.get(run_id_str)
        if span is None:
            return

        if self._chain_types.get(run_id_str) == ChainType.METRIC.value:
            score = outputs.get("output")
            safe_score = _safe_attribute(score)
            if isinstance(safe_score, (int, float)):
                span.set_attribute("ragas.metric.score", safe_score)

        span.end()

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: t.Any,
    ) -> t.Any:
        self._record_error(str(run_id), error)

    def on_llm_start(
        self,
        serialized: t.Dict[str, t.Any],
        prompts: t.List[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: t.Optional[uuid.UUID] = None,
        tags: t.Optional[t.List[str]] = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        **kwargs: t.Any,
    ) -> t.Any:
        self._start_llm_span(serialized, len(prompts), run_id, parent_run_id, metadata)

    def on_chat_model_start(
        self,
        serialized: t.Dict[str, t.Any],
        messages: t.List[t.List[t.Any]],
        *,
        run_id: uuid.UUID,
        parent_run_id: t.Optional[uuid.UUID] = None,
        tags: t.Optional[t.List[str]] = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        **kwargs: t.Any,
    ) -> t.Any:
        self._start_llm_span(serialized, len(messages), run_id, parent_run_id, metadata)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        **kwargs: t.Any,
    ) -> t.Any:
        span = self._spans.get(str(run_id))
        if span is None:
            return

        for key, value in self._llm_attributes(response).items():
            safe_value = _safe_attribute(value)
            if safe_value is not None:
                span.set_attribute(key, safe_value)

        span.end()

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: t.Any,
    ) -> t.Any:
        self._record_error(str(run_id), error)

    def _start_llm_span(
        self,
        serialized: t.Dict[str, t.Any],
        prompt_count: int,
        run_id: uuid.UUID,
        parent_run_id: t.Optional[uuid.UUID],
        metadata: t.Optional[t.Dict[str, t.Any]],
    ) -> None:
        run_id_str = str(run_id)
        parent_id = str(parent_run_id) if parent_run_id else None
        self._parents[run_id_str] = parent_id

        span = self.tracer.start_span(  # type: ignore[union-attr]
            "ragas.llm.call",
            context=self._parent_context(parent_id),
        )
        self._spans[run_id_str] = span
        self._contexts[run_id_str] = trace.set_span_in_context(span)

        span.set_attribute("ragas.run.id", run_id_str)
        if parent_id is not None:
            span.set_attribute("ragas.parent_run.id", parent_id)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("ragas.llm.prompt_count", prompt_count)

        model_name = self._model_name(serialized, metadata)
        if model_name:
            span.set_attribute("gen_ai.request.model", model_name)
            span.set_attribute("llm.model_name", model_name)

        row_index = self._nearest_row_index(parent_id)
        if row_index is not None:
            span.set_attribute("ragas.row.index", row_index)

    def _record_error(self, run_id: str, error: BaseException) -> None:
        span = self._spans.get(run_id)
        if span is None:
            return

        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.end()

    def _span_name(
        self,
        run_name: str,
        metadata: t.Optional[t.Dict[str, t.Any]],
    ) -> t.Optional[str]:
        chain_type = _chain_type_value(metadata)
        if chain_type == ChainType.EVALUATION.value:
            return "ragas.evaluate"
        if chain_type == ChainType.ROW.value:
            return "ragas.row"
        if chain_type == ChainType.METRIC.value:
            return f"ragas.metric.{run_name}"
        return None

    def _set_common_attributes(
        self,
        span: Span,
        serialized: t.Dict[str, t.Any],
        inputs: t.Dict[str, t.Any],
        run_id: str,
        parent_id: t.Optional[str],
        metadata: t.Optional[t.Dict[str, t.Any]],
    ) -> None:
        run_name = serialized.get("name", "")
        chain_type = _chain_type_value(metadata)

        span.set_attribute("ragas.run.id", run_id)
        span.set_attribute("ragas.run.name", run_name)
        if parent_id is not None:
            span.set_attribute("ragas.parent_run.id", parent_id)
        if chain_type is not None:
            span.set_attribute("ragas.chain.type", chain_type)

        if chain_type == ChainType.ROW.value and metadata is not None:
            row_index = metadata.get("row_index")
            if isinstance(row_index, int):
                self._row_indexes[run_id] = row_index
                span.set_attribute("ragas.row.index", row_index)
        elif chain_type == ChainType.METRIC.value:
            span.set_attribute("ragas.metric.name", run_name)
            row_index = self._nearest_row_index(parent_id)
            if row_index is not None:
                span.set_attribute("ragas.row.index", row_index)

        if chain_type == ChainType.EVALUATION.value:
            span.set_attribute("ragas.evaluation.input_keys", list(inputs.keys()))

    def _parent_context(self, parent_id: t.Optional[str]) -> t.Optional[Context]:
        while parent_id is not None:
            parent_context = self._contexts.get(parent_id)
            if parent_context is not None:
                return parent_context
            parent_id = self._parents.get(parent_id)
        return context_api.get_current()

    def _nearest_row_index(self, run_id: t.Optional[str]) -> t.Optional[int]:
        while run_id is not None:
            if run_id in self._row_indexes:
                return self._row_indexes[run_id]
            run_id = self._parents.get(run_id)
        return None

    def _model_name(
        self,
        serialized: t.Dict[str, t.Any],
        metadata: t.Optional[t.Dict[str, t.Any]],
    ) -> t.Optional[str]:
        for mapping in (serialized, metadata or {}):
            model_name = _get_nested(
                mapping,
                [
                    "kwargs.model",
                    "kwargs.model_name",
                    "kwargs.model_id",
                    "model",
                    "model_name",
                    "model_id",
                    "name",
                    "id.0",
                ],
            )
            if isinstance(model_name, str) and model_name:
                return model_name
        return None

    def _llm_attributes(self, response: LLMResult) -> t.Dict[str, t.Any]:
        llm_output = response.llm_output or {}
        response_metadata = _generation_metadata(response)
        usage = _get_nested(
            llm_output,
            [
                "token_usage",
                "usage",
                "usage_metadata",
            ],
        )
        if not isinstance(usage, Mapping):
            usage = _get_nested(
                response_metadata,
                [
                    "token_usage",
                    "usage",
                    "usage_metadata",
                ],
            )

        usage = usage if isinstance(usage, Mapping) else {}
        input_tokens = _get_nested(
            usage,
            ["input_tokens", "prompt_tokens", "input_token_count"],
        )
        output_tokens = _get_nested(
            usage,
            ["output_tokens", "completion_tokens", "output_token_count"],
        )
        total_tokens = _get_nested(usage, ["total_tokens", "total_token_count"])
        model_name = _get_nested(
            llm_output,
            ["model_name", "model", "model_id"],
        ) or _get_nested(response_metadata, ["model_name", "model", "model_id"])

        attributes = {
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
            "llm.usage.total_tokens": total_tokens,
            "gen_ai.response.model": model_name,
        }
        if model_name is not None:
            attributes["llm.model_name"] = model_name
        return attributes


RagasTracer = OpenTelemetryTracer
