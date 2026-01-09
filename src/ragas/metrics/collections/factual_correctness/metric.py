"""Factual Correctness metrics v2 - Modern implementation with multi-modal scoring."""

import typing as t
from typing import List

import numpy as np

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.metrics.utils import fbeta_score

from .util import (
    ClaimDecompositionInput,
    ClaimDecompositionOutput,
    ClaimDecompositionPrompt,
    NLIStatementInput,
    NLIStatementOutput,
    NLIStatementPrompt,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class FactualCorrectness(BaseMetric):
    """
    Modern v2 implementation of factual correctness evaluation.

    Evaluates the factual correctness of responses by comparing claims made in the response
    against a reference text. Uses claim decomposition and natural language inference (NLI)
    to verify claims in both directions.

    The metric supports three evaluation modes:
    - Precision: What fraction of response claims are supported by reference
    - Recall: What fraction of reference claims are covered by response
    - F1: Harmonic mean of precision and recall (with configurable beta)

    The metric also supports configurable claim decomposition:
    - Atomicity: "low" (fewer, broader claims) vs "high" (more, atomic claims)
    - Coverage: "low" (partial coverage) vs "high" (comprehensive coverage)

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import FactualCorrectness
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = FactualCorrectness(llm=llm, mode="f1", beta=1.0)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     response="Einstein was born in Germany in 1879.",
        ...     reference="Albert Einstein was born in Ulm, Germany on March 14, 1879."
        ... )
        >>> print(f"Factual Correctness: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for claim decomposition and NLI evaluation
        mode: Evaluation mode ("precision", "recall", or "f1")
        beta: Beta parameter for F1 score (>1 favors recall, <1 favors precision)
        atomicity: Claim decomposition atomicity ("low" or "high")
        coverage: Claim decomposition coverage ("low" or "high")
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        mode: t.Literal["precision", "recall", "f1"] = "f1",
        beta: float = 1.0,
        atomicity: t.Literal["low", "high"] = "low",
        coverage: t.Literal["low", "high"] = "low",
        name: str = "factual_correctness",
        **kwargs,
    ):
        """
        Initialize FactualCorrectness metric with required components.

        Args:
            llm: Modern instructor-based LLM for claim decomposition and NLI evaluation
            mode: Evaluation mode ("precision", "recall", or "f1")
            beta: Beta parameter for F1 score (>1 favors recall, <1 favors precision)
            atomicity: Claim decomposition atomicity ("low" or "high")
            coverage: Claim decomposition coverage ("low" or "high")
            name: The metric name
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.mode = mode
        self.beta = beta
        self.atomicity = atomicity
        self.coverage = coverage
        self.prompt = ClaimDecompositionPrompt()
        self.nli_prompt = NLIStatementPrompt()

        # Validate beta parameter
        if not isinstance(beta, (int, float)):
            raise ValueError(
                "Beta must be a float. A beta > 1 gives more weight to recall, while beta < 1 favors precision."
            )

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(self, response: str, reference: str) -> MetricResult:
        """
        Calculate factual correctness score.

        Args:
            response: The response to evaluate for factual correctness
            reference: The reference text to check claims against

        Returns:
            MetricResult with factual correctness score (0.0-1.0, higher is better)
        """
        # Input validation
        if not response:
            raise ValueError(
                "response is missing. Please add response to the test sample."
            )
        if not reference:
            raise ValueError(
                "reference is missing. Please add reference to the test sample."
            )

        # Step 1: Get claim verifications to match legacy behavior exactly
        # Legacy always does: decompose response → verify against reference
        reference_response = await self._decompose_and_verify_claims(
            response, reference
        )

        if self.mode != "precision":
            # For recall and f1, also do: decompose reference → verify against response
            response_reference = await self._decompose_and_verify_claims(
                reference, response
            )
        else:
            response_reference = np.array([], dtype=bool)

        # Step 2: Compute TP, FP, FN exactly like legacy
        tp = int(np.sum(reference_response))
        fp = int(np.sum(~reference_response))
        if self.mode != "precision":
            fn = int(np.sum(~response_reference))
        else:
            fn = 0

        # Step 3: Compute final score based on mode
        if self.mode == "precision":
            score = tp / (tp + fp + 1e-8)
        elif self.mode == "recall":
            score = tp / (tp + fn + 1e-8)
        else:  # f1
            score = fbeta_score(tp, fp, fn, self.beta)

        return MetricResult(value=float(np.round(score, 2)))

    async def _decompose_claims(self, text: str) -> List[str]:
        """Break text into claims using configurable decomposition."""
        input_data = ClaimDecompositionInput(
            response=text, atomicity=self.atomicity, coverage=self.coverage
        )
        prompt_str = self.prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_str, ClaimDecompositionOutput)
        return result.claims

    async def _verify_claims(
        self, claims: List[str], reference: str
    ) -> NLIStatementOutput:
        """Verify claims against reference using NLI."""
        input_data = NLIStatementInput(context=reference, statements=claims)
        prompt_str = self.nli_prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_str, NLIStatementOutput)
        return result

    async def _decompose_and_verify_claims(
        self, text_to_decompose: str, reference_text: str
    ) -> np.ndarray:
        """Decompose text into claims and verify against reference."""
        claims = await self._decompose_claims(text_to_decompose)
        if not claims:
            return np.array([], dtype=bool)

        verdicts = await self._verify_claims(claims, reference_text)
        if not verdicts.statements:
            return np.array([], dtype=bool)

        return np.array([bool(stmt.verdict) for stmt in verdicts.statements])
