"""Context Entity Recall metrics v2 - Modern implementation with structured prompts."""

import typing as t
from typing import List, Sequence

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import (
    EntitiesList,
    ExtractEntitiesInput,
    ExtractEntitiesPrompt,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class ContextEntityRecall(BaseMetric):
    """
    Modern v2 implementation of context entity recall evaluation.

    Calculates recall based on entities present in ground truth and retrieved contexts.
    Let CN be the set of entities present in context,
    GN be the set of entities present in the ground truth.
    Context Entity recall = | CN ∩ GN | / | GN |

    This implementation uses modern instructor LLMs with structured output.
    Only supports modern components - legacy wrappers are rejected with clear error messages.

    Usage:
        >>> import openai
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ContextEntityRecall
        >>>
        >>> # Setup dependencies
        >>> client = openai.AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = ContextEntityRecall(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     reference="Paris is the capital of France, established in 52 BC.",
        ...     retrieved_contexts=["France's capital city is Paris.", "The city was founded in ancient times."]
        ... )
        >>> print(f"Entity Recall: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for entity extraction
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "context_entity_recall",
        **kwargs,
    ):
        """
        Initialize ContextEntityRecall metric with required components.

        Args:
            llm: Modern instructor-based LLM for entity extraction
            name: The metric name (default: "context_entity_recall")
            **kwargs: Additional arguments passed to BaseMetric
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.prompt = ExtractEntitiesPrompt()  # Initialize prompt class once

        # Call super() for validation
        super().__init__(name=name, **kwargs)

    async def ascore(
        self, reference: str, retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Calculate context entity recall score.

        Components are guaranteed to be validated and non-None by the base class.

        Args:
            reference: The ground truth reference text
            retrieved_contexts: List of retrieved context strings

        Returns:
            MetricResult with entity recall score (0.0-1.0, higher is better)
        """
        # Extract entities from reference (ground truth)
        reference_entities = await self._extract_entities(reference)

        # Extract entities from retrieved contexts (combined)
        combined_contexts = "\n".join(retrieved_contexts)
        context_entities = await self._extract_entities(combined_contexts)

        # Calculate recall score
        recall_score = self._compute_recall_score(reference_entities, context_entities)

        return MetricResult(value=float(recall_score))

    async def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text using the entity extraction prompt.

        Args:
            text: The text to extract entities from

        Returns:
            List of extracted entities
        """
        # Create input data and generate prompt
        input_data = ExtractEntitiesInput(text=text)
        prompt_string = self.prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_string, EntitiesList)
        return result.entities

    def _compute_recall_score(
        self, reference_entities: Sequence[str], context_entities: Sequence[str]
    ) -> float:
        """
        Compute entity recall score using set intersection.

        Recall = |intersection| / |reference|

        Args:
            reference_entities: Entities from the reference text
            context_entities: Entities from the context

        Returns:
            Entity recall score (0.0-1.0)
        """
        reference_set = set(reference_entities)
        context_set = set(context_entities)

        # Calculate intersection
        entities_in_both = len(reference_set.intersection(context_set))

        # Calculate recall: |intersection| / |reference|
        # Add small epsilon to avoid division by zero
        recall = entities_in_both / (len(reference_set) + 1e-8)

        return recall
