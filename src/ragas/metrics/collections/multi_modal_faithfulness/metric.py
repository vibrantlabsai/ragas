"""MultiModalFaithfulness metric - Collections implementation for multimodal faithfulness evaluation."""

import typing as t
from typing import List

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import (
    MULTIMODAL_FAITHFULNESS_INSTRUCTION,
    MultiModalFaithfulnessOutput,
    build_multimodal_message_content,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorLLM


class MultiModalFaithfulness(BaseMetric):
    """
    MultiModalFaithfulness metric for evaluating response faithfulness against
    both visual and textual context.

    Measures how factually consistent a response is with the retrieved context,
    which can include both text and images. A response is considered faithful
    if all its claims can be supported by the provided contexts.

    The metric returns a binary score:
    - 1.0 if the response is faithful to the contexts
    - 0.0 if the response is not faithful

    This implementation uses modern instructor LLMs with vision capabilities
    for multimodal evaluation.

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import MultiModalFaithfulness
        >>>
        >>> # Setup dependencies (use a vision-capable model)
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o", client=client)  # Vision-capable model
        >>>
        >>> # Create metric instance
        >>> metric = MultiModalFaithfulness(llm=llm)
        >>>
        >>> # Single evaluation with image context
        >>> result = await metric.ascore(
        ...     response="The Tesla Model X is an electric SUV.",
        ...     retrieved_contexts=["path/to/tesla_image.jpg", "Tesla makes electric vehicles."]
        ... )
        >>> print(f"Faithfulness Score: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM with vision capabilities
        name: The metric name
        allowed_values: Score range (0.0 or 1.0)

    Note:
        This metric requires a vision-capable LLM (e.g., gpt-4o, gpt-4-vision,
        claude-3-opus, gemini-pro-vision) to evaluate image contexts.
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorLLM"

    def __init__(
        self,
        llm: "InstructorLLM",
        name: str = "multi_modal_faithfulness",
        **kwargs,
    ):
        """
        Initialize MultiModalFaithfulness metric with required components.

        Args:
            llm: Modern instructor-based LLM with vision capabilities
            name: The metric name
        """
        self.llm = llm
        super().__init__(name=name, **kwargs)

    async def ascore(
        self,
        response: str,
        retrieved_contexts: List[str],
    ) -> MetricResult:
        """
        Calculate multimodal faithfulness score.

        Args:
            response: The response to evaluate for faithfulness
            retrieved_contexts: List of retrieved contexts (text strings or
                              image paths/URLs/base64 data)

        Returns:
            MetricResult with faithfulness score (0.0 or 1.0)

        Raises:
            ValueError: If response or retrieved_contexts is missing
        """
        # Input validation
        if not response:
            raise ValueError(
                "response is missing. Please provide a response to evaluate."
            )
        if not retrieved_contexts:
            raise ValueError(
                "retrieved_contexts is missing. Please provide contexts to check against."
            )

        # Build multimodal message content
        message_content = build_multimodal_message_content(
            instruction=MULTIMODAL_FAITHFULNESS_INSTRUCTION,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )

        # Call the LLM with multimodal content
        result = await self._evaluate_faithfulness(message_content)

        # Return score based on faithfulness verdict
        score = 1.0 if result.faithful else 0.0
        return MetricResult(value=score, reason=result.reason)

    async def _evaluate_faithfulness(
        self,
        message_content: List[t.Dict[str, t.Any]],
    ) -> MultiModalFaithfulnessOutput:
        """
        Evaluate faithfulness using the LLM with multimodal content.

        Args:
            message_content: List of content blocks (text and images)

        Returns:
            MultiModalFaithfulnessOutput with verdict and reason
        """
        # Build the messages for the LLM
        messages = [{"role": "user", "content": message_content}]

        # Get provider-specific kwargs
        provider_kwargs = self.llm._map_provider_params()

        # Call the LLM directly with multimodal messages
        if self.llm.provider.lower() == "google":
            result = await self.llm.client.create(
                messages=messages,
                response_model=MultiModalFaithfulnessOutput,
                **provider_kwargs,
            )
        else:
            result = await self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                response_model=MultiModalFaithfulnessOutput,
                **provider_kwargs,
            )

        return result
