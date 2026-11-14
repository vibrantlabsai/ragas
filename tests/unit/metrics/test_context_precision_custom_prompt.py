"""Tests for configurable Context Precision prompts."""

from unittest.mock import patch

import pytest

from ragas.metrics.collections.context_precision.metric import (
    ContextPrecision,
    ContextPrecisionWithoutReference,
    ContextPrecisionWithReference,
    ContextUtilization,
)
from ragas.metrics.collections.context_precision.util import ContextPrecisionPrompt


@pytest.mark.parametrize(
    "metric_class",
    [
        ContextPrecision,
        ContextPrecisionWithReference,
        ContextPrecisionWithoutReference,
        ContextUtilization,
    ],
)
def test_context_precision_metrics_accept_custom_prompt(metric_class):
    prompt = ContextPrecisionPrompt()
    # These tests exercise prompt wiring, not LLM component validation.
    with patch("ragas.metrics.collections.base.BaseMetric._validate_llm"):
        metric = metric_class(llm=object(), prompt=prompt)

    assert metric.prompt is prompt


def test_context_precision_metrics_create_default_prompt():
    with patch("ragas.metrics.collections.base.BaseMetric._validate_llm"):
        metric = ContextPrecision(llm=object())

    assert isinstance(metric.prompt, ContextPrecisionPrompt)
