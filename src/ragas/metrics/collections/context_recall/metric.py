"""Context Recall metrics v2 - Modern implementation with structured prompts."""

import typing as t
from typing import List

import numpy as np

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import (
    ContextRecallInput,
    ContextRecallOutput,
    ContextRecallPrompt,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class ContextRecall(BaseMetric):
    """
    Modern v2 implementation of context recall evaluation.

    Evaluates context recall by classifying if statements in a reference answer
    can be attributed to the retrieved context. Uses an LLM to verify attribution
    for each statement and calculates recall as the proportion of attributed statements.

    This implementation uses modern instructor LLMs with structured output.
    Only supports modern components - legacy wrappers are rejected with clear error messages.

    Usage:
        >>> import openai
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ContextRecall
        >>>
        >>> # Setup dependencies
        >>> client = openai.AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = ContextRecall(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     retrieved_contexts=["Paris is the capital of France."],
        ...     reference="Paris is the capital and largest city of France."
        ... )
        >>> print(f"Context Recall: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for statement classification
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "context_recall",
        **kwargs,
    ):
        """
        Initialize ContextRecall metric with required components.

        Args:
            llm: Modern instructor-based LLM for statement classification
            name: The metric name (default: "context_recall")
            **kwargs: Additional arguments passed to BaseMetric
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.prompt = ContextRecallPrompt()  # Initialize prompt class once

        # Call super() for validation
        super().__init__(name=name, **kwargs)

    async def ascore(
        self,
        user_input: str,
        retrieved_contexts: List[str],
        reference: str,
    ) -> MetricResult:
        """
        Calculate context recall score asynchronously.

        Components are guaranteed to be validated and non-None by the base class.

        Args:
            user_input: The original question
            retrieved_contexts: List of retrieved context strings
            reference: The reference answer to evaluate

        Returns:
            MetricResult with recall score (0.0-1.0, higher is better)
        """
        # Input validation
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not reference:
            raise ValueError("reference cannot be empty")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts cannot be empty")

        # Combine contexts into a single string
        context = "\n".join(retrieved_contexts) if retrieved_contexts else ""

        # Create input data and generate prompt
        input_data = ContextRecallInput(
            question=user_input, context=context, answer=reference
        )
        prompt_string = self.prompt.to_string(input_data)

        # Get classifications from LLM
        result = await self.llm.agenerate(prompt_string, ContextRecallOutput)

        # Calculate score
        if not result.classifications:
            return MetricResult(value=np.nan)

        # Count attributions
        attributions = [c.attributed for c in result.classifications]
        score = sum(attributions) / len(attributions) if attributions else np.nan

        return MetricResult(value=float(score))
