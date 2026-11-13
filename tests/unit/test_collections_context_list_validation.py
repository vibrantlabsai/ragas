"""Collections metrics must reject a string where a list of contexts is expected.

A string is iterable, so before this check a stringified list such as "['a', 'b']" (what
the local/csv backend returns after a reload) was scored one character at a time: no
error, a meaningless score, and for ContextPrecision one judge call per character.
"""

import inspect

import pytest

import ragas.metrics.collections as collections
from ragas.backends.local_csv import LocalCSVBackend
from ragas.llms.base import InstructorLLM
from ragas.metrics.collections.base import validate_text_list

CONTEXTS = ["Paris is the capital of France.", "Paris is in Europe."]
STRINGIFIED = str(CONTEXTS)  # what local/csv hands back for a list column

INPUTS = {
    "user_input": "What is the capital of France?",
    "response": "Paris is the capital of France.",
    "reference": "Paris",
    "rubrics": {"score1_description": "wrong", "score5_description": "right"},
}

# metric class -> name of its context-list argument
METRICS = {
    "ContextEntityRecall": "retrieved_contexts",
    "ContextPrecision": "retrieved_contexts",
    "ContextPrecisionWithReference": "retrieved_contexts",
    "ContextPrecisionWithoutReference": "retrieved_contexts",
    "ContextRecall": "retrieved_contexts",
    "ContextRelevance": "retrieved_contexts",
    "ContextUtilization": "retrieved_contexts",
    "Faithfulness": "retrieved_contexts",
    "MultiModalFaithfulness": "retrieved_contexts",
    "MultiModalRelevance": "retrieved_contexts",
    "NoiseSensitivity": "retrieved_contexts",
    "ResponseGroundedness": "retrieved_contexts",
    "SQLSemanticEquivalence": "reference_contexts",
    "SummaryScore": "reference_contexts",
}

# Metrics that take a context list but are protected some other way.
EXEMPT = {
    # Validates its input model with pydantic before calling the judge.
    "DomainSpecificRubrics",
    "InstanceSpecificRubrics",
    "RubricsScoreWithReference",
    "RubricsScoreWithoutReference",
    # Non-LLM metric; already returns "Invalid input" for a non-list.
    "QuotedSpansAlignment",
}


class JudgeReached(Exception):
    """Raised by the fake provider so a test can see the judge was called."""


class CountingProvider:
    """Stands in for an OpenAI client and counts every completion request."""

    def __init__(self):
        self.calls = 0
        provider = self

        class _Completions:
            async def create(self, *args, **kwargs):
                provider.calls += 1
                raise JudgeReached()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()
        self.client = self


def _metric(name):
    provider = CountingProvider()
    llm = InstructorLLM(client=provider, model="fake-judge", provider="openai")
    return getattr(collections, name)(llm=llm), provider


def _kwargs(metric, contexts_arg, contexts):
    params = inspect.signature(metric.ascore).parameters
    kwargs = {k: v for k, v in INPUTS.items() if k in params}
    kwargs[contexts_arg] = contexts
    return kwargs


def _context_metrics():
    names = set()
    for name in dir(collections):
        cls = getattr(collections, name)
        if inspect.isclass(cls) and hasattr(cls, "ascore"):
            params = inspect.signature(getattr(cls, "ascore")).parameters
            if any(p in params for p in ("retrieved_contexts", "reference_contexts")):
                names.add(name)
    return names


def test_every_context_metric_is_covered():
    """A new metric that takes contexts must be added to METRICS (or EXEMPT)."""
    assert _context_metrics() == set(METRICS) | EXEMPT


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(METRICS))
@pytest.mark.parametrize("bad", [STRINGIFIED, "Paris is the capital of France."])
async def test_string_contexts_fail_before_any_judge_call(name, bad):
    metric, provider = _metric(name)
    with pytest.raises(ValueError, match="must be a list of strings"):
        await metric.ascore(**_kwargs(metric, METRICS[name], bad))
    assert provider.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(METRICS | dict.fromkeys(EXEMPT, None)))
async def test_list_contexts_still_reach_the_judge(name):
    """Valid input passes the check unchanged and reaches the judge.

    The fake judge raises; some metrics catch judge errors themselves, so only the
    call count is asserted.
    """
    if name == "QuotedSpansAlignment":
        pytest.skip("non-LLM metric")
    metric, provider = _metric(name)
    arg = METRICS.get(name) or "retrieved_contexts"
    try:
        await metric.ascore(**_kwargs(metric, arg, CONTEXTS))
    except JudgeReached:
        pass
    assert provider.calls >= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(EXEMPT - {"QuotedSpansAlignment"}))
async def test_exempt_metrics_already_reject_string_contexts(name):
    metric, provider = _metric(name)
    with pytest.raises(ValueError):
        await metric.ascore(**_kwargs(metric, "retrieved_contexts", STRINGIFIED))
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_context_precision_scores_a_list_as_before():
    """One judge call per context, same score as before the check was added."""
    provider = CountingProvider()

    async def useful(*args, response_model, **kwargs):
        provider.calls += 1
        return response_model(reason="useful", verdict=1)

    provider.chat.completions.create = useful
    llm = InstructorLLM(client=provider, model="fake-judge", provider="openai")
    result = await collections.ContextPrecision(llm=llm).ascore(
        user_input=INPUTS["user_input"],
        reference=INPUTS["reference"],
        retrieved_contexts=CONTEXTS,
    )
    assert provider.calls == len(CONTEXTS)
    assert result.value == pytest.approx(1.0)


def test_error_explains_a_stringified_list():
    with pytest.raises(ValueError, match="local/csv"):
        validate_text_list("retrieved_contexts", STRINGIFIED)
    with pytest.raises(ValueError, match="wrap it in a list"):
        validate_text_list("retrieved_contexts", "one context")


@pytest.mark.parametrize("value", [CONTEXTS, tuple(CONTEXTS), [], None])
def test_non_string_values_are_left_to_the_metric(value):
    """Lists, tuples, and the existing empty/None checks are unchanged."""
    validate_text_list("retrieved_contexts", value)


@pytest.mark.asyncio
async def test_context_list_reloaded_from_local_csv_is_rejected(tmp_path):
    """The real path: save a row to local/csv, load it back, score it."""
    backend = LocalCSVBackend(root_dir=str(tmp_path))
    backend.save_dataset("rag", [{**INPUTS, "retrieved_contexts": CONTEXTS}])
    row = backend.load_dataset("rag")[0]
    assert row["retrieved_contexts"] == STRINGIFIED  # documented: values become text

    metric, provider = _metric("ContextPrecision")
    with pytest.raises(ValueError, match="local/csv"):
        await metric.ascore(
            user_input=row["user_input"],
            reference=row["reference"],
            retrieved_contexts=row["retrieved_contexts"],
        )
    assert provider.calls == 0
