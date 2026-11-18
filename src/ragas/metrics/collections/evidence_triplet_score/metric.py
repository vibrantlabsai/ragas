"""Evidence triplet score metric - Modern collections implementation."""

import typing as t

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import EvidenceTripletInput, EvidenceTripletOutput, EvidenceTripletPrompt

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class EvidenceTripletScore(BaseMetric):
    """
    Evaluate RAG evidence as truth, indeterminacy, and falsity.

    This metric evaluates a generated response against retrieved contexts and
    returns a neutrosophic evidence triplet:
    - truth: evidence supports the response
    - indeterminacy: evidence is missing, incomplete, ambiguous, or insufficient
    - falsity: evidence contradicts the response or exposes unsupported claims

    The triplet dimensions are independent and should not be normalized to sum
    to 1.0. This preserves the neutrosophic interpretation introduced by
    Florentin Smarandache while keeping the metric opt-in and compatible with
    Ragas' collections API.

    Usage:
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import EvidenceTripletScore
        >>>
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>> metric = EvidenceTripletScore(llm=llm)
        >>> result = await metric.ascore(
        ...     user_input="What caused the outage?",
        ...     response="The outage was caused by a database failover.",
        ...     retrieved_contexts=[
        ...         "Network routing errors were detected during the outage.",
        ...         "Root cause analysis is still pending.",
        ...     ],
        ... )
        >>> print(result.value)

    Attributes:
        llm: Modern instructor-based LLM used for structured evidence scoring.
        name: Metric name.
    """

    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "evidence_triplet_score",
        **kwargs,
    ):
        self.llm = llm
        self.scoring_prompt = EvidenceTripletPrompt()
        super().__init__(name=name, **kwargs)

    async def ascore(
        self,
        user_input: str,
        response: str,
        retrieved_contexts: t.List[str],
    ) -> MetricResult:
        """
        Calculate a neutrosophic evidence triplet.

        Args:
            user_input: The question or input provided to the RAG system.
            response: The generated response to evaluate.
            retrieved_contexts: Context passages retrieved for the response.

        Returns:
            MetricResult whose value is a dict with truth, indeterminacy, and
            falsity scores in [0.0, 1.0], and whose reason explains the scoring.
        """
        self._validate_inputs(user_input, response, retrieved_contexts)

        prompt_input = EvidenceTripletInput(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        prompt_str = self.scoring_prompt.to_string(prompt_input)
        result: EvidenceTripletOutput = await self.llm.agenerate(
            prompt_str, EvidenceTripletOutput
        )
        self._validate_output(result)

        return MetricResult(
            value={
                "truth": float(result.truth),
                "indeterminacy": float(result.indeterminacy),
                "falsity": float(result.falsity),
            },
            reason=result.reason,
        )

    def _validate_inputs(
        self,
        user_input: str,
        response: str,
        retrieved_contexts: t.List[str],
    ) -> None:
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input is missing. Please add user_input.")
        if not isinstance(response, str) or not response.strip():
            raise ValueError("response is missing. Please add response.")
        if not isinstance(retrieved_contexts, list) or not retrieved_contexts:
            raise ValueError(
                "retrieved_contexts is missing. Please add retrieved_contexts."
            )
        if not all(isinstance(context, str) for context in retrieved_contexts):
            raise ValueError("retrieved_contexts must be a list of strings.")
        if not any(context.strip() for context in retrieved_contexts):
            raise ValueError(
                "retrieved_contexts must contain at least one non-empty context."
            )

    def _validate_output(self, output: EvidenceTripletOutput) -> None:
        values = {
            "truth": output.truth,
            "indeterminacy": output.indeterminacy,
            "falsity": output.falsity,
        }
        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0.0, 1.0], got {value!r}.")
        if not output.reason.strip():
            raise ValueError("reason is missing from evidence triplet output.")


class NeutrosophicEvidenceTriplet(EvidenceTripletScore):
    """
    Academic-name alias for EvidenceTripletScore.

    This alias preserves the neutrosophic origin of the T/I/F model while
    keeping `EvidenceTripletScore` available as a product-facing metric name.
    """

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "neutrosophic_evidence_triplet",
        **kwargs,
    ):
        super().__init__(llm=llm, name=name, **kwargs)
