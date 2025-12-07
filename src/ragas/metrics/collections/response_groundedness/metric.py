"""Response Groundedness metric v2 - Modern implementation with dual-judge evaluation."""

import typing as t
from typing import List

import numpy as np

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import (
    ResponseGroundednessInput,
    ResponseGroundednessJudge1Prompt,
    ResponseGroundednessJudge2Prompt,
    ResponseGroundednessOutput,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class ResponseGroundedness(BaseMetric):
    """
    Response Groundedness metric using dual-judge evaluation.

    Evaluates how well grounded a response is in the retrieved contexts
    using a dual-judge system. This metric averages two distinct judge prompts
    to ensure robust evaluation.

    The metric uses NVIDIA's proven dual-judge approach:
    1. Judge 1: Direct groundedness evaluation with structured instructions
    2. Judge 2: Alternative perspective for fairness
    3. Average both judges for final score

    Rating scale: 0 (not grounded), 1 (partially grounded), 2 (fully grounded)
    Final score: Average of both judges converted to 0.0-1.0 scale

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ResponseGroundedness
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = ResponseGroundedness(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     response="Einstein was born in Germany in 1879.",
        ...     retrieved_contexts=["Albert Einstein was born in Ulm, Germany on March 14, 1879."]
        ... )
        >>> print(f"Response Groundedness: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for dual-judge evaluation
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
        max_retries: Maximum retry attempts for invalid ratings
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "response_groundedness",
        max_retries: int = 5,
        **kwargs,
    ):
        """
        Initialize ResponseGroundedness metric with required components.

        Args:
            llm: Modern instructor-based LLM for dual-judge evaluation
            name: The metric name
            max_retries: Maximum retry attempts for invalid ratings
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.max_retries = max_retries
        self.judge1_prompt = ResponseGroundednessJudge1Prompt()
        self.judge2_prompt = ResponseGroundednessJudge2Prompt()

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(
        self, response: str, retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Calculate response groundedness score using dual-judge evaluation.

        Args:
            response: The response to evaluate for groundedness
            retrieved_contexts: The retrieved contexts to check groundedness against

        Returns:
            MetricResult with response groundedness score (0.0-1.0, higher is better)
        """
        # Input validation
        if not response:
            raise ValueError(
                "response is missing. Please add response to the test sample."
            )
        if not retrieved_contexts:
            raise ValueError(
                "retrieved_contexts is missing. Please add retrieved_contexts to the test sample."
            )

        # Handle edge cases like legacy
        context_str = "\n".join(retrieved_contexts)

        if not response.strip() or not context_str.strip():
            return MetricResult(value=0.0)

        # Get ratings from both judges
        judge1_rating = await self._get_judge_rating(
            self.judge1_prompt, response, context_str
        )
        judge2_rating = await self._get_judge_rating(
            self.judge2_prompt, response, context_str
        )

        # Average the scores (convert from 0,1,2 scale to 0.0-1.0)
        score = self._average_scores(judge1_rating / 2.0, judge2_rating / 2.0)

        return MetricResult(value=float(score))

    async def _get_judge_rating(self, prompt_obj, response: str, context: str) -> float:
        """Get rating from judge with retry logic."""
        for retry in range(self.max_retries):
            try:
                input_data = ResponseGroundednessInput(
                    response=response, context=context
                )
                prompt_str = prompt_obj.to_string(input_data)
                result = await self.llm.agenerate(
                    prompt_str, ResponseGroundednessOutput
                )
                rating = result.rating

                # Validate rating is in expected range
                if rating in [0, 1, 2]:
                    return float(rating)
                else:
                    if retry < self.max_retries - 1:
                        continue  # Retry if invalid rating
                    else:
                        return float("nan")

            except Exception:
                if retry < self.max_retries - 1:
                    continue  # Retry on exception
                else:
                    return float("nan")

        return float("nan")

    def _average_scores(self, score1: float, score2: float) -> float:
        """Average two judge scores, handling NaN values."""
        if not np.isnan(score1) and not np.isnan(score2):
            return (score1 + score2) / 2.0
        elif not np.isnan(score1):
            return score1
        elif not np.isnan(score2):
            return score2
        else:
            return float("nan")
