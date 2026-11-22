"""Tests for CitationFaithfulness metric (collections implementation).

These tests never call a real LLM. The pure helpers in ``util.py`` are tested in
isolation (like ``quoted_spans/util.py``), and the metric is driven by a mock
InstructorLLM whose ``agenerate`` dispatches on the requested response model
(claim extraction first, attribution verification second).
"""

import math

import pytest

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import CitationFaithfulness
from ragas.metrics.collections.citation_faithfulness.util import (
    CitationAttributionPrompt,
    CitationAttributionVerdict,
    CitationClaimInput,
    CitationClaimOutput,
    CitationClaimPrompt,
    CitationVerdictInput,
    CitationVerdictItem,
    CitationVerdictOutput,
    CitedClaim,
    compute_citation_score,
    has_citation_markers,
    parse_citation_markers,
)

# Two retrieved passages used across the attribution tests. Passage [1] supports
# the "capital of France" claim; passage [2] supports the "Eiffel Tower" claim.
CONTEXTS = [
    "Paris is the capital of France.",
    "The Eiffel Tower is located in Paris.",
]


class MockInstructorLLM(InstructorBaseRagasLLM):
    """Mock InstructorLLM that returns fixed structured outputs per response model.

    The metric calls ``agenerate`` twice: once with ``CitationClaimOutput`` (step
    1, claim extraction) and once with ``CitationVerdictOutput`` (step 2,
    attribution). This mock dispatches on the requested model and records every
    call so tests can assert what was sent.
    """

    def __init__(self, claim_output=None, verdict_output=None):
        self._claim_output = claim_output
        self._verdict_output = verdict_output
        self.calls = []

    def generate(self, prompt, response_model):
        raise NotImplementedError("sync generate is not used in these tests")

    async def agenerate(self, prompt, response_model):
        self.calls.append((prompt, response_model))
        if response_model is CitationClaimOutput:
            return self._claim_output
        if response_model is CitationVerdictOutput:
            return self._verdict_output
        raise AssertionError(f"Unexpected response_model: {response_model!r}")


def _verdict(claim, citation, value):
    return CitationAttributionVerdict(
        claim=claim, citation=citation, reason="stub", verdict=value
    )


# --------------------------------------------------------------------------- #
# Pure helper tests (no LLM)
# --------------------------------------------------------------------------- #


class TestCitationHelpers:
    def test_has_citation_markers_present(self):
        assert has_citation_markers("Paris is the capital of France [1].") is True

    def test_has_citation_markers_spaced(self):
        assert has_citation_markers("A cited claim [ 12 ].") is True

    def test_has_citation_markers_grouped(self):
        assert has_citation_markers("A cited claim [1, 2].") is True

    def test_has_citation_markers_absent(self):
        assert has_citation_markers("No markers here at all.") is False

    def test_has_citation_markers_empty(self):
        assert has_citation_markers("") is False

    def test_parse_citation_markers_multiple(self):
        assert parse_citation_markers("Claim one [1]. Claim two [2].") == [1, 2]

    def test_parse_citation_markers_grouped(self):
        assert parse_citation_markers("A claim [1, 2] here.") == [1, 2]

    def test_parse_citation_markers_none(self):
        assert parse_citation_markers("No markers here.") == []

    def test_parse_citation_markers_empty_string(self):
        assert parse_citation_markers("") == []

    def test_compute_score_no_claims_is_nan(self):
        assert math.isnan(compute_citation_score([], 2, {}))

    def test_compute_score_all_correct(self):
        score = compute_citation_score([[1], [2]], 2, {(0, 1): True, (1, 2): True})
        assert score == 1.0

    def test_compute_score_one_misattributed(self):
        # Both claims cite [2]; only the second is actually supported by [2].
        score = compute_citation_score([[2], [2]], 2, {(0, 2): False, (1, 2): True})
        assert score == pytest.approx(0.5)

    def test_compute_score_out_of_range_is_incorrect(self):
        # Claim 1 cites [3], which is out of range and never in the verdict lookup.
        score = compute_citation_score([[1], [3]], 2, {(0, 1): True})
        assert score == pytest.approx(0.5)

    def test_compute_score_multi_citation_all_correct(self):
        score = compute_citation_score([[1, 2]], 2, {(0, 1): True, (0, 2): True})
        assert score == 1.0

    def test_compute_score_multi_citation_one_wrong(self):
        # A claim is correct only if ALL its citations are correct.
        score = compute_citation_score([[1, 2]], 2, {(0, 1): True, (0, 2): False})
        assert score == 0.0

    def test_compute_score_empty_citation_entry_is_incorrect(self):
        score = compute_citation_score([[1], []], 2, {(0, 1): True})
        assert score == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Prompt / model round-trip tests (no LLM)
# --------------------------------------------------------------------------- #


class TestCitationPrompts:
    def test_claim_prompt_attributes(self):
        prompt = CitationClaimPrompt()
        assert prompt.input_model is CitationClaimInput
        assert prompt.output_model is CitationClaimOutput
        assert len(prompt.examples) >= 1
        rendered = prompt.to_string(CitationClaimInput(question="q?", answer="a [1]."))
        assert isinstance(rendered, str)
        assert prompt.instruction in rendered

    def test_attribution_prompt_forbids_supported_elsewhere(self):
        prompt = CitationAttributionPrompt()
        assert prompt.input_model is CitationVerdictInput
        assert prompt.output_model is CitationVerdictOutput
        # The instruction must explicitly refuse "supported elsewhere" credit.
        assert "some other passage" in prompt.instruction.lower()
        rendered = prompt.to_string(
            CitationVerdictInput(
                items=[
                    CitationVerdictItem(
                        claim="Paris is the capital of France.",
                        citation=1,
                        context=CONTEXTS[0],
                    )
                ]
            )
        )
        assert isinstance(rendered, str)

    def test_models_round_trip(self):
        claim_out = CitationClaimOutput(
            claims=[CitedClaim(claim="Paris is the capital of France.", citations=[1])]
        )
        assert CitationClaimOutput.model_validate(claim_out.model_dump()) == claim_out

        verdict_out = CitationVerdictOutput(
            verdicts=[_verdict("Paris is the capital of France.", 1, 1)]
        )
        assert (
            CitationVerdictOutput.model_validate(verdict_out.model_dump())
            == verdict_out
        )


# --------------------------------------------------------------------------- #
# Metric tests (mocked LLM)
# --------------------------------------------------------------------------- #


class TestCitationFaithfulnessMetric:
    def test_init_defaults(self):
        metric = CitationFaithfulness(llm=MockInstructorLLM())
        assert metric.name == "citation_faithfulness"
        assert isinstance(metric.claim_extraction_prompt, CitationClaimPrompt)
        assert isinstance(metric.attribution_prompt, CitationAttributionPrompt)

    def test_custom_name(self):
        metric = CitationFaithfulness(llm=MockInstructorLLM(), name="my_citation")
        assert metric.name == "my_citation"

    @pytest.mark.asyncio
    async def test_correct_attribution_scores_one(self):
        """Every claim cites the passage that actually supports it -> 1.0."""
        claim_output = CitationClaimOutput(
            claims=[
                CitedClaim(claim="Paris is the capital of France.", citations=[1]),
                CitedClaim(
                    claim="The Eiffel Tower is located in Paris.", citations=[2]
                ),
            ]
        )
        verdict_output = CitationVerdictOutput(
            verdicts=[
                _verdict("Paris is the capital of France.", 1, 1),
                _verdict("The Eiffel Tower is located in Paris.", 2, 1),
            ]
        )
        metric = CitationFaithfulness(
            llm=MockInstructorLLM(claim_output, verdict_output)
        )

        result = await metric.ascore(
            user_input="Tell me about Paris.",
            response="Paris is the capital of France [1]. The Eiffel Tower is in Paris [2].",
            retrieved_contexts=CONTEXTS,
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_supported_elsewhere_misattribution_scores_below_one(self):
        """A claim cited to a passage that does not support it scores < 1.0.

        The 'capital of France' claim is cited to passage [2] (the Eiffel Tower
        passage). Passage [1] does support it, so plain faithfulness would pass
        this answer. Citation faithfulness must flag the misattribution.
        """
        claim_output = CitationClaimOutput(
            claims=[
                CitedClaim(claim="Paris is the capital of France.", citations=[2]),
                CitedClaim(
                    claim="The Eiffel Tower is located in Paris.", citations=[2]
                ),
            ]
        )
        # Items are sent in claim order: (claim0, [2]) then (claim1, [2]).
        verdict_output = CitationVerdictOutput(
            verdicts=[
                _verdict("Paris is the capital of France.", 2, 0),
                _verdict("The Eiffel Tower is located in Paris.", 2, 1),
            ]
        )
        metric = CitationFaithfulness(
            llm=MockInstructorLLM(claim_output, verdict_output)
        )

        result = await metric.ascore(
            user_input="Tell me about Paris.",
            response="Paris is the capital of France [2]. The Eiffel Tower is in Paris [2].",
            retrieved_contexts=CONTEXTS,
        )
        assert result.value == pytest.approx(0.5)
        assert result.value < 1.0

    @pytest.mark.asyncio
    async def test_out_of_range_citation_is_incorrect(self):
        """A citation index pointing at a non-retrieved passage counts wrong."""
        claim_output = CitationClaimOutput(
            claims=[
                CitedClaim(claim="Paris is the capital of France.", citations=[1]),
                CitedClaim(
                    claim="The Eiffel Tower is located in Paris.", citations=[3]
                ),
            ]
        )
        # Only the in-range citation [1] is sent to the LLM. [3] never is.
        verdict_output = CitationVerdictOutput(
            verdicts=[_verdict("Paris is the capital of France.", 1, 1)]
        )
        llm = MockInstructorLLM(claim_output, verdict_output)
        metric = CitationFaithfulness(llm=llm)

        result = await metric.ascore(
            user_input="Tell me about Paris.",
            response="Paris is the capital of France [1]. The Eiffel Tower is in Paris [3].",
            retrieved_contexts=CONTEXTS,
        )
        assert result.value == pytest.approx(0.5)

        # Confirm the out-of-range citation was never sent to the verdict step.
        verdict_calls = [c for c in llm.calls if c[1] is CitationVerdictOutput]
        assert len(verdict_calls) == 1
        sent_prompt = verdict_calls[0][0]
        assert '"citation": 3' not in sent_prompt

    @pytest.mark.asyncio
    async def test_no_markers_returns_nan(self):
        """A response with no [N] markers is a degenerate case -> NaN.

        The claim stub would return a claim, but the fast path short-circuits
        before the LLM is ever called.
        """
        llm = MockInstructorLLM(
            CitationClaimOutput(
                claims=[
                    CitedClaim(claim="Paris is the capital of France.", citations=[1])
                ]
            )
        )
        metric = CitationFaithfulness(llm=llm)

        result = await metric.ascore(
            user_input="Tell me about Paris.",
            response="Paris is the capital of France. The Eiffel Tower is in Paris.",
            retrieved_contexts=CONTEXTS,
        )
        assert isinstance(result.value, float)
        assert math.isnan(result.value)
        # Fast path means no LLM call at all.
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_no_cited_claims_extracted_returns_nan(self):
        """Markers exist but the extractor finds no cited claims -> NaN."""
        llm = MockInstructorLLM(CitationClaimOutput(claims=[]))
        metric = CitationFaithfulness(llm=llm)

        result = await metric.ascore(
            user_input="Tell me about Paris.",
            response="Paris is the capital of France [1].",
            retrieved_contexts=CONTEXTS,
        )
        assert math.isnan(result.value)
        # Step 1 ran; step 2 (verdict) never did.
        assert all(c[1] is CitationClaimOutput for c in llm.calls)

    @pytest.mark.asyncio
    async def test_claims_without_citations_are_dropped(self):
        """Extracted claims lacking citations are ignored; only cited ones score."""
        claim_output = CitationClaimOutput(
            claims=[
                CitedClaim(claim="Paris is the capital of France.", citations=[1]),
                CitedClaim(claim="Paris is a nice city.", citations=[]),
            ]
        )
        verdict_output = CitationVerdictOutput(
            verdicts=[_verdict("Paris is the capital of France.", 1, 1)]
        )
        metric = CitationFaithfulness(
            llm=MockInstructorLLM(claim_output, verdict_output)
        )

        result = await metric.ascore(
            user_input="Tell me about Paris.",
            response="Paris is the capital of France [1]. Paris is a nice city.",
            retrieved_contexts=CONTEXTS,
        )
        # Only the single cited claim counts, and it is correct.
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_missing_response_raises(self):
        metric = CitationFaithfulness(llm=MockInstructorLLM())
        with pytest.raises(ValueError, match="response is missing"):
            await metric.ascore(
                user_input="q", response="", retrieved_contexts=CONTEXTS
            )

    @pytest.mark.asyncio
    async def test_missing_user_input_raises(self):
        metric = CitationFaithfulness(llm=MockInstructorLLM())
        with pytest.raises(ValueError, match="user_input is missing"):
            await metric.ascore(
                user_input="", response="Paris [1].", retrieved_contexts=CONTEXTS
            )

    @pytest.mark.asyncio
    async def test_missing_retrieved_contexts_raises(self):
        metric = CitationFaithfulness(llm=MockInstructorLLM())
        with pytest.raises(ValueError, match="retrieved_contexts is missing"):
            await metric.ascore(
                user_input="q", response="Paris [1].", retrieved_contexts=[]
            )

    def test_sync_score_method(self):
        claim_output = CitationClaimOutput(
            claims=[
                CitedClaim(claim="Paris is the capital of France.", citations=[1]),
            ]
        )
        verdict_output = CitationVerdictOutput(
            verdicts=[_verdict("Paris is the capital of France.", 1, 1)]
        )
        metric = CitationFaithfulness(
            llm=MockInstructorLLM(claim_output, verdict_output)
        )

        result = metric.score(
            user_input="Tell me about Paris.",
            response="Paris is the capital of France [1].",
            retrieved_contexts=CONTEXTS,
        )
        assert result.value == 1.0
