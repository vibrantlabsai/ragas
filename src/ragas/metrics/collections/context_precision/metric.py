"""Context Precision metrics v2 - Modern implementation with function-based prompts."""

import typing as t
from typing import List

import numpy as np

from ragas.decision_models import BaseDecisionModel, BinaryDecisionResult
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import (
    ContextPrecisionInput,
    ContextPrecisionOutput,
    ContextPrecisionPrompt,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class ContextPrecisionWithReference(BaseMetric):
    """
    Modern v2 implementation of context precision with reference.

    Evaluates whether retrieved contexts are useful for answering a question by comparing
    each context against a reference answer. The metric calculates average precision
    based on the usefulness verdicts from an LLM.

    Uses a modern structured-output LLM by default. An explicit decision_model
    instead evaluates each context without requiring an LLM. Its selected binary
    verdicts feed the same average precision calculation.

    Usage:
        >>> import openai
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ContextPrecisionWithReference
        >>>
        >>> # Setup dependencies
        >>> client = openai.AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = ContextPrecisionWithReference(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     reference="Paris is the capital of France.",
        ...     retrieved_contexts=["Paris is the capital and largest city of France.", "Berlin is the capital of Germany."]
        ... )
        >>> print(f"Context Precision: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for context evaluation
        decision_model: Optional binary judge, taking precedence over llm
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: t.Optional["InstructorBaseRagasLLM"]

    def __init__(
        self,
        llm: t.Optional["InstructorBaseRagasLLM"] = None,
        name: str = "context_precision_with_reference",
        *,
        decision_model: t.Optional[BaseDecisionModel] = None,
        **kwargs,
    ):
        """
        Initialize ContextPrecisionWithReference metric with required components.

        Args:
            llm: Modern instructor-based LLM for context evaluation
            name: The metric name
            decision_model: Optional binary judge. Receives question, context,
                and answer fields. Its value is used without rethresholding.
                Per-context results are retained in result.traces["output"]["decisions"].
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        if decision_model is not None and not isinstance(
            decision_model, BaseDecisionModel
        ):
            raise TypeError("decision_model must be a BaseDecisionModel")
        self.decision_model = decision_model
        self.prompt = ContextPrecisionPrompt()  # Initialize prompt class once

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

    def _validate_llm(self):
        if self.llm is None and self.decision_model is not None:
            return
        super()._validate_llm()

    async def ascore(
        self, user_input: str, reference: str, retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Calculate context precision score using reference.

        Args:
            user_input: The question being asked
            reference: The reference answer to compare against
            retrieved_contexts: The retrieved contexts to evaluate

        Returns:
            MetricResult with context precision score (0.0-1.0, higher is better)
        """
        # Input validation
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not reference:
            raise ValueError("reference cannot be empty")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts cannot be empty")

        # Evaluate each retrieved context
        verdicts = []
        decisions = []
        for context in retrieved_contexts:
            # Create input data and generate prompt
            input_data = ContextPrecisionInput(
                question=user_input, context=context, answer=reference
            )
            if self.decision_model is not None:
                decision = await self.decision_model.binary(
                    instruction=(
                        "Given question, answer and context, decide whether the context "
                        "was useful in arriving at the given answer."
                    ),
                    data=input_data.model_dump(),
                )
                if not isinstance(decision, BinaryDecisionResult):
                    raise TypeError(
                        "decision_model.binary() must return BinaryDecisionResult"
                    )
                decision = BinaryDecisionResult.model_validate(decision)
                verdicts.append(int(decision.value))
                decisions.append(decision.model_dump())
            else:
                assert self.llm is not None
                prompt_string = self.prompt.to_string(input_data)
                result = await self.llm.agenerate(prompt_string, ContextPrecisionOutput)
                verdicts.append(result.verdict)

        # Calculate average precision
        score = self._calculate_average_precision(verdicts)
        if self.decision_model is not None:
            return MetricResult(
                value=float(score), traces={"output": {"decisions": decisions}}
            )
        return MetricResult(value=float(score))

    def _calculate_average_precision(self, verdicts: List[int]) -> float:
        """Calculate average precision from binary verdicts."""
        cumsum = 0
        numerator = 0.0
        for i, v in enumerate(verdicts):
            cumsum += v
            if v:
                numerator += cumsum / (i + 1)

        denominator = cumsum + 1e-10
        score = numerator / denominator

        if np.isnan(score):
            # Match legacy warning behavior
            import logging

            logging.warning(
                "Invalid response format. Expected a list of dictionaries with keys 'verdict'"
            )

        return score


class ContextPrecisionWithoutReference(BaseMetric):
    """
    Modern v2 implementation of context precision without reference.

    Evaluates whether retrieved contexts are useful for answering a question by comparing
    each context against the generated response. The metric calculates average precision
    based on the usefulness verdicts from an LLM.

    This implementation uses modern instructor LLMs with structured output.
    Only supports modern components - legacy wrappers are rejected with clear error messages.

    Usage:
        >>> import openai
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ContextPrecisionWithoutReference
        >>>
        >>> # Setup dependencies
        >>> client = openai.AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = ContextPrecisionWithoutReference(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     response="Paris is the capital of France.",
        ...     retrieved_contexts=["Paris is the capital and largest city of France.", "Berlin is the capital of Germany."]
        ... )
        >>> print(f"Context Precision: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for context evaluation
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "context_precision_without_reference",
        **kwargs,
    ):
        """
        Initialize ContextPrecisionWithoutReference metric with required components.

        Args:
            llm: Modern instructor-based LLM for context evaluation
            name: The metric name
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.prompt = ContextPrecisionPrompt()  # Initialize prompt class once

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(
        self, user_input: str, response: str, retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Calculate context precision score using response.

        Args:
            user_input: The question being asked
            response: The response that was generated
            retrieved_contexts: The retrieved contexts to evaluate

        Returns:
            MetricResult with context precision score (0.0-1.0, higher is better)
        """
        # Input validation
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not response:
            raise ValueError("response cannot be empty")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts cannot be empty")

        # Evaluate each retrieved context
        verdicts = []
        for context in retrieved_contexts:
            # Create input data and generate prompt
            input_data = ContextPrecisionInput(
                question=user_input, context=context, answer=response
            )
            prompt_string = self.prompt.to_string(input_data)
            result = await self.llm.agenerate(prompt_string, ContextPrecisionOutput)
            verdicts.append(result.verdict)

        # Calculate average precision
        score = self._calculate_average_precision(verdicts)
        return MetricResult(value=float(score))

    def _calculate_average_precision(self, verdicts: List[int]) -> float:
        """Calculate average precision from binary verdicts."""
        cumsum = 0
        numerator = 0.0
        for i, v in enumerate(verdicts):
            cumsum += v
            if v:
                numerator += cumsum / (i + 1)

        denominator = cumsum + 1e-10
        score = numerator / denominator

        if np.isnan(score):
            # Match legacy warning behavior
            import logging

            logging.warning(
                "Invalid response format. Expected a list of dictionaries with keys 'verdict'"
            )

        return score


class ContextPrecision(ContextPrecisionWithReference):
    """
    Modern v2 wrapper for ContextPrecisionWithReference with shorter name.

    This is a simple wrapper that provides the legacy "context_precision" name
    while using the modern V2 implementation underneath.

    Usage:
        >>> import openai
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ContextPrecision
        >>>
        >>> # Setup dependencies
        >>> client = openai.AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> # Create metric instance (same as ContextPrecisionWithReference)
        >>> metric = ContextPrecision(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     reference="Paris is the capital of France.",
        ...     retrieved_contexts=["Paris is the capital and largest city of France."]
        ... )
    """

    def __init__(
        self,
        llm: t.Optional["InstructorBaseRagasLLM"] = None,
        *,
        decision_model: t.Optional[BaseDecisionModel] = None,
        **kwargs,
    ):
        """Initialize ContextPrecision with the legacy default name."""
        super().__init__(
            llm, name="context_precision", decision_model=decision_model, **kwargs
        )


class ContextUtilization(ContextPrecisionWithoutReference):
    """
    Modern v2 wrapper for ContextPrecisionWithoutReference with shorter name.

    This is a simple wrapper that provides the legacy "context_utilization" name
    while using the modern V2 implementation underneath.

    Usage:
        >>> import openai
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ContextUtilization
        >>>
        >>> # Setup dependencies
        >>> client = openai.AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> # Create metric instance (same as ContextPrecisionWithoutReference)
        >>> metric = ContextUtilization(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     response="Paris is the capital of France.",
        ...     retrieved_contexts=["Paris is the capital and largest city of France."]
        ... )
    """

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        **kwargs,
    ):
        """Initialize ContextUtilization with the legacy default name."""
        super().__init__(llm, name="context_utilization", **kwargs)
