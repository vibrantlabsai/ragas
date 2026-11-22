"""Citation faithfulness metric - Modern collections implementation."""

import logging
import typing as t
from typing import Dict, List, Tuple

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import (
    CitationAttributionPrompt,
    CitationClaimInput,
    CitationClaimOutput,
    CitationClaimPrompt,
    CitationVerdictInput,
    CitationVerdictItem,
    CitationVerdictOutput,
    CitedClaim,
    compute_citation_score,
    has_citation_markers,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM

logger = logging.getLogger(__name__)


class CitationFaithfulness(BaseMetric):
    """
    Citation-attribution faithfulness using a two-step LLM pipeline.

    Measures whether each inline ``[N]`` citation marker in the response points to
    the specific retrieved passage that supports the claim it is attached to. This
    differs from Faithfulness, which only checks whether a claim is supported by
    the retrieved contexts somewhere. The failure mode this metric catches is
    "supported-elsewhere misattribution": a claim cited to passage ``[A]`` that
    ``[A]`` does not support, even though another retrieved passage ``[B]`` does.

    The metric works by:
    1. Breaking the response into claims, each carrying its cited passage indices
       (parsed from the ``[N]`` markers by the LLM).
    2. Verifying, for each cited claim, that every cited passage supports that
       specific claim.
    3. Scoring as the fraction of cited claims whose citations are all correct.

    Indexing convention: markers are 1-based, so ``[1]`` refers to the first
    retrieved context (``retrieved_contexts[0]``). A citation index that is out of
    range (points to a passage that was not retrieved) counts as an incorrect
    citation and is never sent to the LLM.

    Degenerate case: if the response contains no citation markers, or no cited
    claims are extracted, the metric is undefined and returns ``float("nan")``
    (mirroring how Faithfulness returns NaN when no statements are generated).

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import CitationFaithfulness
        >>>
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> metric = CitationFaithfulness(llm=llm)
        >>>
        >>> result = await metric.ascore(
        ...     user_input="Tell me about Paris.",
        ...     response="Paris is the capital of France [1].",
        ...     retrieved_contexts=[
        ...         "Paris is the capital of France.",
        ...         "The Eiffel Tower is located in Paris.",
        ...     ],
        ... )
        >>> print(f"Citation Faithfulness: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for claim extraction and attribution.
        name: The metric name (default: "citation_faithfulness").
        allowed_values: Score range (0.0 to 1.0, higher is better).
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "citation_faithfulness",
        **kwargs,
    ):
        """
        Initialize CitationFaithfulness metric with required components.

        Args:
            llm: Modern instructor-based LLM for claim extraction and attribution.
            name: The metric name.
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.claim_extraction_prompt = CitationClaimPrompt()
        self.attribution_prompt = CitationAttributionPrompt()

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(
        self, user_input: str, response: str, retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Calculate citation-attribution faithfulness using a two-step pipeline.

        Args:
            user_input: The original question.
            response: The response to evaluate, containing ``[N]`` citation markers.
            retrieved_contexts: The retrieved contexts the citations index into.

        Returns:
            MetricResult with citation faithfulness score (0.0-1.0, higher is
            better), or NaN for the degenerate no-citation case.
        """
        # Input validation (mirrors collections/faithfulness)
        if not response:
            raise ValueError(
                "response is missing. Please add response to the test sample."
            )
        if not user_input:
            raise ValueError(
                "user_input is missing. Please add user_input to the test sample."
            )
        if not retrieved_contexts:
            raise ValueError(
                "retrieved_contexts is missing. Please add retrieved_contexts to the test sample."
            )

        # Fast path for the degenerate no-citation case: undefined, like
        # Faithfulness returns NaN when no statements are generated.
        if not has_citation_markers(response):
            logger.warning(
                "No citation markers ([N]) found in the response; "
                "CitationFaithfulness is undefined for this sample."
            )
            return MetricResult(value=float("nan"))

        # Step 1: Break response into claims, each with its cited passage indices.
        cited_claims = await self._extract_cited_claims(user_input, response)
        cited_claims = [c for c in cited_claims if c.citations]

        if not cited_claims:
            logger.warning(
                "No cited claims were extracted from the response; "
                "CitationFaithfulness is undefined for this sample."
            )
            return MetricResult(value=float("nan"))

        # Step 2: Verify each cited passage supports the claim it is attached to.
        verdicts = await self._verify_citations(cited_claims, retrieved_contexts)

        # Step 3: Compute score as fraction of fully correctly attributed claims.
        claim_citations = [c.citations for c in cited_claims]
        score = compute_citation_score(
            claim_citations, len(retrieved_contexts), verdicts
        )

        return MetricResult(value=float(score))

    async def _extract_cited_claims(
        self, question: str, response: str
    ) -> List[CitedClaim]:
        """Break the response into claims with their cited passage indices."""
        input_data = CitationClaimInput(question=question, answer=response)
        prompt_str = self.claim_extraction_prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_str, CitationClaimOutput)
        return result.claims

    async def _verify_citations(
        self, cited_claims: List[CitedClaim], contexts: List[str]
    ) -> Dict[Tuple[int, int], bool]:
        """
        Verify each in-range citation against the passage it is attached to.

        Returns a lookup of (claim_index, citation) -> whether that specific
        passage supports the claim. Out-of-range citations are handled by the
        score computation and are never sent to the LLM.
        """
        num_contexts = len(contexts)
        items: List[CitationVerdictItem] = []
        item_refs: List[Tuple[int, int]] = []
        for claim_idx, cited_claim in enumerate(cited_claims):
            for citation in cited_claim.citations:
                if 1 <= citation <= num_contexts:
                    items.append(
                        CitationVerdictItem(
                            claim=cited_claim.claim,
                            citation=citation,
                            context=contexts[citation - 1],
                        )
                    )
                    item_refs.append((claim_idx, citation))

        verdicts: Dict[Tuple[int, int], bool] = {}
        if items:
            prompt_str = self.attribution_prompt.to_string(
                CitationVerdictInput(items=items)
            )
            output = await self.llm.agenerate(prompt_str, CitationVerdictOutput)
            # Align verdicts to the items we sent, by position (as Faithfulness
            # aligns its NLI verdicts).
            for ref, verdict in zip(item_refs, output.verdicts):
                verdicts[ref] = bool(verdict.verdict)
            # Any item the LLM did not return a verdict for is treated as False.
            for ref in item_refs:
                verdicts.setdefault(ref, False)

        return verdicts
