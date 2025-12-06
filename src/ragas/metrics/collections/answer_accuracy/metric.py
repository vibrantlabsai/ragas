"""Answer Accuracy metric v2 - Modern implementation with dual-judge evaluation."""

import typing as t

import numpy as np

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import (
    AnswerAccuracyInput,
    AnswerAccuracyJudge1Prompt,
    AnswerAccuracyJudge2Prompt,
    AnswerAccuracyOutput,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class AnswerAccuracy(BaseMetric):
    """
    Answer Accuracy metric using dual-judge evaluation.

    Measures answer accuracy compared to ground truth using a dual-judge system.
    This metric averages two distinct judge prompts to ensure robust evaluation.

    The metric uses NVIDIA's proven dual-judge approach:
    1. Judge 1: Direct User Answer vs Reference Answer comparison
    2. Judge 2: Swapped perspective for fairness
    3. Average both judges for final score

    Rating scale: 0 (no match), 2 (partial match), 4 (exact match)
    Final score: Average of both judges converted to 0.0-1.0 scale

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import AnswerAccuracy
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = AnswerAccuracy(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="When was Einstein born?",
        ...     response="Albert Einstein was born in 1879.",
        ...     reference="Albert Einstein was born in 1879."
        ... )
        >>> print(f"Answer Accuracy: {result.value}")

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
        name: str = "answer_accuracy",
        max_retries: int = 5,
        **kwargs,
    ):
        """
        Initialize AnswerAccuracy metric with required components.

        Args:
            llm: Modern instructor-based LLM for dual-judge evaluation
            name: The metric name
            max_retries: Maximum retry attempts for invalid ratings
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.max_retries = max_retries
        self.judge1_prompt = AnswerAccuracyJudge1Prompt()
        self.judge2_prompt = AnswerAccuracyJudge2Prompt()

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(
        self, user_input: str, response: str, reference: str
    ) -> MetricResult:
        """
        Calculate answer accuracy score using dual-judge evaluation.

        Args:
            user_input: The original question
            response: The user's answer to evaluate
            reference: The ground truth reference answer

        Returns:
            MetricResult with answer accuracy score (0.0-1.0, higher is better)
        """
        # Input validation
        if not user_input:
            raise ValueError(
                "user_input is missing. Please add user_input to the test sample."
            )
        if not response:
            raise ValueError(
                "response is missing. Please add response to the test sample."
            )
        if not reference:
            raise ValueError(
                "reference is missing. Please add reference to the test sample."
            )

        # Get ratings from both judges
        judge1_rating = await self._get_judge_rating(
            self.judge1_prompt, user_input, response, reference
        )
        judge2_rating = await self._get_judge_rating(
            self.judge2_prompt, user_input, reference, response
        )  # Note: swapped order for judge 2

        # Average the scores (convert from 0,2,4 scale to 0.0-1.0)
        score = self._average_scores(judge1_rating / 4.0, judge2_rating / 4.0)

        return MetricResult(value=float(score))

    async def _get_judge_rating(
        self, prompt_obj, query: str, user_answer: str, reference_answer: str
    ) -> float:
        """Get rating from judge with retry logic."""
        for retry in range(self.max_retries):
            try:
                input_data = AnswerAccuracyInput(
                    query=query,
                    user_answer=user_answer,
                    reference_answer=reference_answer,
                )
                prompt_str = prompt_obj.to_string(input_data)
                result = await self.llm.agenerate(prompt_str, AnswerAccuracyOutput)
                rating = result.rating

                # Validate rating is in expected range
                if rating in [0, 2, 4]:
                    return float(rating)
                else:
                    # Invalid rating - retry or return NaN
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
