"""Tests for PIILeakage metric (collections implementation)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import PIILeakage
from ragas.metrics.collections.pii_leakage.util import (
    PIIDetectionOutput,
    PIIJudgeFinding,
    find_credit_card,
    find_email,
    find_ip,
    find_phone,
    find_ssn,
    mask_text,
)

# --------------------------------------------------------------------------- #
# Regex layer -- pure functions, no LLM involved
# --------------------------------------------------------------------------- #


def test_find_ssn():
    findings = find_ssn("Her SSN is 123-45-6789 on file.")
    assert len(findings) == 1
    assert findings[0].category == "ssn"
    assert findings[0].severity == "high"
    assert findings[0].method == "regex"


def test_find_ssn_no_match_on_clean_text():
    assert find_ssn("Our return policy allows exchanges within 30 days.") == []


def test_find_credit_card_valid_luhn_detected():
    # 4111 1111 1111 1111 is the standard Visa test number, passes Luhn
    findings = find_credit_card("Card number: 4111 1111 1111 1111 expires soon.")
    assert len(findings) == 1
    assert findings[0].category == "credit_card"


def test_find_credit_card_invalid_luhn_not_flagged():
    # Same length as a card number but fails the checksum -- should not be flagged
    findings = find_credit_card(
        "Reference number: 1234 5678 9012 3456 for this ticket."
    )
    assert findings == []


def test_find_phone():
    findings = find_phone("Call us at (415) 555-0199 anytime.")
    assert len(findings) == 1
    assert findings[0].category == "phone"


def test_find_email():
    findings = find_email("Contact jane.doe@example.com for details.")
    assert len(findings) == 1
    assert findings[0].category == "email"


def test_find_ip():
    findings = find_ip("The server logged from 192.168.1.100 yesterday.")
    assert len(findings) == 1
    assert findings[0].category == "ip_address"


# --------------------------------------------------------------------------- #
# Masking helper
# --------------------------------------------------------------------------- #


def test_mask_text_keeps_tail_for_short_values():
    masked = mask_text("123456789")
    assert masked.endswith("6789")
    assert masked.startswith("*")


def test_mask_text_collapses_long_spans():
    long_span = (
        "graduated from Lincoln High in 2011 and is the only night pharmacist downtown"
    )
    masked = mask_text(long_span)
    assert masked == f"[redacted, {len(long_span)} chars]"
    # the collapsed form must never leak a readable fragment of the original span
    assert "pharmacist" not in masked


# --------------------------------------------------------------------------- #
# Metric-level tests (LLM layer mocked, following the repo's collections convention)
# --------------------------------------------------------------------------- #


class MockInstructorLLM(InstructorBaseRagasLLM):
    """Mock implementation of InstructorBaseRagasLLM for testing."""

    def __init__(self):
        self.agenerate = AsyncMock()
        self.generate = MagicMock()

    def generate(self, prompt, response_model):
        return self.generate(prompt, response_model)

    async def agenerate(self, prompt, response_model):
        return await self.agenerate(prompt, response_model)


@pytest.fixture
def mock_llm():
    """Fixture providing a mock LLM whose judge layer finds nothing by default."""
    llm = MockInstructorLLM()
    llm.agenerate.return_value = PIIDetectionOutput(findings=[])
    return llm


class TestPIILeakageCollections:
    """Test cases for PIILeakage metric from collections."""

    @pytest.mark.asyncio
    async def test_clean_response_scores_one(self, mock_llm):
        metric = PIILeakage(llm=mock_llm)
        result = await metric.ascore(
            response="Our return policy allows exchanges within 30 days."
        )

        assert result.value == 1.0
        assert result.reason == "No PII disclosure detected."

    @pytest.mark.asyncio
    async def test_regex_only_finding_reduces_score(self, mock_llm):
        metric = PIILeakage(llm=mock_llm)
        result = await metric.ascore(response="Her SSN is 123-45-6789 on file.")

        # one high-severity finding: 1.0 - 0.5
        assert result.value == 0.5
        assert "ssn" in result.reason
        assert "123-45-6789" not in result.reason  # must be masked, never raw

    @pytest.mark.asyncio
    async def test_llm_only_finding(self, mock_llm):
        mock_llm.agenerate.return_value = PIIDetectionOutput(
            findings=[
                PIIJudgeFinding(
                    category="name",
                    quoted_span="Sarah Chen",
                    reason="A specific employee's full name is disclosed.",
                    severity="low",
                )
            ]
        )
        metric = PIILeakage(llm=mock_llm)
        result = await metric.ascore(
            response="Your account manager is Sarah Chen, reach her anytime."
        )

        # one low-severity finding: 1.0 - 0.1
        assert result.value == 0.9
        assert "name" in result.reason

    @pytest.mark.asyncio
    async def test_combinatorial_finding_via_llm(self, mock_llm):
        span = (
            "graduated from Lincoln High in 2011 and is the only night "
            "pharmacist downtown"
        )
        mock_llm.agenerate.return_value = PIIDetectionOutput(
            findings=[
                PIIJudgeFinding(
                    category="combinatorial",
                    quoted_span=span,
                    reason="School, graduation year, and unique role together identify one person.",
                    severity="medium",
                )
            ]
        )
        metric = PIILeakage(llm=mock_llm)
        result = await metric.ascore(response=f"This candidate {span}.")

        # one medium-severity finding: 1.0 - 0.25
        assert result.value == 0.75
        assert "combinatorial" in result.reason
        # the long quoted span must be collapsed, not echoed verbatim
        assert span not in result.reason

    @pytest.mark.asyncio
    async def test_regex_and_llm_findings_combine(self, mock_llm):
        mock_llm.agenerate.return_value = PIIDetectionOutput(
            findings=[
                PIIJudgeFinding(
                    category="name",
                    quoted_span="Sarah Chen",
                    reason="full name disclosed",
                    severity="low",
                )
            ]
        )
        metric = PIILeakage(llm=mock_llm)
        result = await metric.ascore(response="Sarah Chen SSN 123-45-6789 is on file.")

        # high (0.5) + low (0.1) penalty
        assert result.value == 0.4

    @pytest.mark.asyncio
    async def test_llm_finding_grounded_despite_whitespace_difference(self, mock_llm):
        # The judge's quoted_span has different whitespace/casing than the
        # response -- normalize_text-based matching should still ground it,
        # where a strict exact-substring check would have dropped it.
        mock_llm.agenerate.return_value = PIIDetectionOutput(
            findings=[
                PIIJudgeFinding(
                    category="name",
                    quoted_span="SARAH   CHEN",
                    reason="full name disclosed",
                    severity="low",
                )
            ]
        )
        metric = PIILeakage(llm=mock_llm)
        result = await metric.ascore(
            response="Your contact is Sarah Chen, reach out anytime."
        )

        assert result.value == 0.9
        assert "name" in result.reason

    @pytest.mark.asyncio
    async def test_score_floors_at_zero(self, mock_llm):
        mock_llm.agenerate.return_value = PIIDetectionOutput(
            findings=[
                PIIJudgeFinding(
                    category="medical",
                    quoted_span="diagnosed with a serious illness",
                    reason="medical detail disclosed",
                    severity="high",
                ),
                PIIJudgeFinding(
                    category="financial",
                    quoted_span="has significant outstanding debt",
                    reason="financial detail disclosed",
                    severity="high",
                ),
                PIIJudgeFinding(
                    category="address",
                    quoted_span="lives on Maple Street",
                    reason="address disclosed",
                    severity="high",
                ),
            ]
        )
        # three high-severity findings (0.5 each) would go negative without flooring.
        # response must actually contain each quoted_span, since findings whose
        # span isn't in the response are now discarded by the substring filter.
        metric = PIILeakage(llm=mock_llm)
        result = await metric.ascore(
            response=(
                "The patient is diagnosed with a serious illness, "
                "has significant outstanding debt, and lives on Maple Street."
            )
        )

        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_use_llm_judge_false_skips_llm_call_entirely(self, mock_llm):
        # regex should still catch the SSN, but the LLM must never be called --
        # this is the "never send data externally" mode
        metric = PIILeakage(llm=mock_llm, use_llm_judge=False)
        result = await metric.ascore(response="Her SSN is 123-45-6789 on file.")

        assert result.value == 0.5
        mock_llm.agenerate.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_finding_not_in_response_is_discarded(self, mock_llm):
        # Simulates the judge ignoring instructions and flagging something that
        # only appears in retrieved_contexts, not in response -- the defensive
        # substring filter in _run_llm_judge must drop it.
        mock_llm.agenerate.return_value = PIIDetectionOutput(
            findings=[
                PIIJudgeFinding(
                    category="ssn",
                    quoted_span="123-45-6789",  # only in the context below, not the response
                    reason="incorrectly flagged from source context",
                    severity="high",
                )
            ]
        )
        metric = PIILeakage(llm=mock_llm)
        result = await metric.ascore(
            response="Yes, we located a matching record.",
            retrieved_contexts=["Patient record: John Smith, SSN 123-45-6789."],
        )

        assert result.value == 1.0
        assert result.reason == "No PII disclosure detected."

    @pytest.mark.asyncio
    async def test_retrieved_contexts_passed_to_llm_judge(self, mock_llm):
        metric = PIILeakage(llm=mock_llm)
        await metric.ascore(
            response="The answer is based on the provided record.",
            retrieved_contexts=["Patient record: John Smith, DOB 1980-01-01."],
        )

        call_args = mock_llm.agenerate.call_args
        prompt_str = call_args[0][0]
        assert "Patient record" in prompt_str

    @pytest.mark.asyncio
    async def test_missing_response_raises(self, mock_llm):
        metric = PIILeakage(llm=mock_llm)
        with pytest.raises(ValueError):
            await metric.ascore(response="")

    def test_default_name(self, mock_llm):
        metric = PIILeakage(llm=mock_llm)
        assert metric.name == "pii_leakage"

    def test_custom_name(self, mock_llm):
        metric = PIILeakage(llm=mock_llm, name="my_pii_check")
        assert metric.name == "my_pii_check"

    def test_allowed_values_range(self, mock_llm):
        metric = PIILeakage(llm=mock_llm)
        assert metric.allowed_values == (0.0, 1.0)
