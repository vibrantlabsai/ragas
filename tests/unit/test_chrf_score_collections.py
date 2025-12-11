"""Tests for CHRFScore metric (collections implementation)."""

import pytest

try:
    from sacrebleu import corpus_chrf  # noqa: F401
except ImportError:
    pytest.skip("sacrebleu not available", allow_module_level=True)

from ragas.metrics.collections import CHRFScore


class TestCHRFScoreCollections:
    """Test cases for CHRFScore metric from collections."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        metric = CHRFScore()
        assert metric.name == "chrf_score"
        assert metric.kwargs == {}

    def test_init_custom_name(self):
        """Test initialization with custom name."""
        metric = CHRFScore(name="custom_chrf")
        assert metric.name == "custom_chrf"

    def test_init_with_kwargs(self):
        """Test initialization with sacrebleu kwargs."""
        metric = CHRFScore(kwargs={"char_order": 4, "word_order": 2})
        assert metric.kwargs == {"char_order": 4, "word_order": 2}

    @pytest.mark.asyncio
    async def test_perfect_match(self):
        """Test perfect match scenario."""
        metric = CHRFScore()

        reference = "The quick brown fox jumps over the lazy dog."
        response = "The quick brown fox jumps over the lazy dog."

        result = await metric.ascore(reference=reference, response=response)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_partial_match(self):
        """Test partial match returns score between 0 and 1."""
        metric = CHRFScore()

        reference = "The quick brown fox jumps over the lazy dog."
        response = "A fast brown fox leaps over a sleepy dog."

        result = await metric.ascore(reference=reference, response=response)
        assert 0.0 < result.value < 1.0

    @pytest.mark.asyncio
    async def test_no_match(self):
        """Test completely different texts."""
        metric = CHRFScore()

        reference = "The quick brown fox jumps over the lazy dog."
        response = "123456789 xyz abc"

        result = await metric.ascore(reference=reference, response=response)
        # Should be low but not necessarily 0 due to character n-gram overlap
        assert result.value < 0.5

    @pytest.mark.asyncio
    async def test_empty_reference(self):
        """Test with empty reference string."""
        metric = CHRFScore()

        result = await metric.ascore(reference="", response="Some text")
        assert result.value == 0.0
        assert "Empty input" in result.reason

    @pytest.mark.asyncio
    async def test_empty_response(self):
        """Test with empty response string."""
        metric = CHRFScore()

        result = await metric.ascore(reference="Some text", response="")
        assert result.value == 0.0
        assert "Empty input" in result.reason

    @pytest.mark.asyncio
    async def test_whitespace_only_input(self):
        """Test with whitespace-only strings."""
        metric = CHRFScore()

        result = await metric.ascore(reference="   ", response="Some text")
        assert result.value == 0.0
        assert "Empty input" in result.reason

    @pytest.mark.asyncio
    async def test_invalid_reference_type(self):
        """Test that non-string reference returns 0.0."""
        metric = CHRFScore()

        result = await metric.ascore(reference=123, response="text")
        assert result.value == 0.0
        assert "Invalid input" in result.reason

    @pytest.mark.asyncio
    async def test_invalid_response_type(self):
        """Test that non-string response returns 0.0."""
        metric = CHRFScore()

        result = await metric.ascore(reference="text", response=456)
        assert result.value == 0.0
        assert "Invalid input" in result.reason

    @pytest.mark.asyncio
    async def test_similar_texts(self):
        """Test similar texts with minor differences."""
        metric = CHRFScore()

        reference = "The capital of France is Paris."
        response = "Paris is the capital of France."

        result = await metric.ascore(reference=reference, response=response)
        # Same words, different order - should have high CHRF score
        assert result.value > 0.6

    @pytest.mark.asyncio
    async def test_score_is_between_0_and_1(self):
        """Test that score is always between 0 and 1."""
        metric = CHRFScore()

        reference = "Machine translation quality assessment."
        response = "Assessment of translation quality for machines."

        result = await metric.ascore(reference=reference, response=response)
        assert 0.0 <= result.value <= 1.0

    def test_sync_score_method(self):
        """Test synchronous score method."""
        metric = CHRFScore()

        reference = "The quick brown fox."
        response = "The quick brown fox."

        result = metric.score(reference=reference, response=response)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_unicode_text(self):
        """Test with unicode characters."""
        metric = CHRFScore()

        reference = "日本語のテスト文字列です。"
        response = "日本語のテスト文字列です。"

        result = await metric.ascore(reference=reference, response=response)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_mixed_case(self):
        """Test case sensitivity handling."""
        metric = CHRFScore()

        reference = "Hello World"
        response = "hello world"

        result = await metric.ascore(reference=reference, response=response)
        # CHRF is case-sensitive, so lowercase version should have lower score
        assert result.value < 1.0
        assert result.value > 0.0  # But still has some similarity

    @pytest.mark.asyncio
    async def test_with_beta_parameter(self):
        """Test with custom beta parameter via kwargs."""
        metric = CHRFScore(kwargs={"beta": 3})

        reference = "The quick brown fox."
        response = "The quick brown fox."

        result = await metric.ascore(reference=reference, response=response)
        assert result.value == 1.0
