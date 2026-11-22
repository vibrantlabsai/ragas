"""Tests for BleuScore metric (collections implementation)."""

import pytest

try:
    from sacrebleu import corpus_bleu  # noqa: F401
except ImportError:
    pytest.skip("sacrebleu not available", allow_module_level=True)

from ragas.metrics.collections import BleuScore


class TestBleuScoreCollections:
    """Test cases for BleuScore metric from collections."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        metric = BleuScore()
        assert metric.name == "bleu_score"
        assert metric.kwargs == {}

    def test_init_custom_name(self):
        """Test initialization with custom name."""
        metric = BleuScore(name="custom_bleu")
        assert metric.name == "custom_bleu"

    def test_init_with_kwargs(self):
        """Test initialization with sacrebleu kwargs."""
        metric = BleuScore(kwargs={"smooth_method": "exp"})
        assert metric.kwargs == {"smooth_method": "exp"}

    @pytest.mark.asyncio
    async def test_perfect_match(self):
        """Test perfect match scenario returns 1.0."""
        metric = BleuScore()

        reference = "The quick brown fox jumps over the lazy dog."
        response = "The quick brown fox jumps over the lazy dog."

        result = await metric.ascore(reference=reference, response=response)
        assert result.value == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_score_is_between_0_and_1(self):
        """Test that score is always between 0 and 1."""
        metric = BleuScore()

        reference = "Machine translation quality assessment."
        response = "Assessment of translation quality for machines."

        result = await metric.ascore(reference=reference, response=response)
        assert 0.0 <= result.value <= 1.0

    @pytest.mark.asyncio
    async def test_evaluates_all_sentences(self):
        """Regression test for #2888: multi-sentence text must be fully evaluated.

        The old sentence-splitting logic caused sacrebleu to truncate to the
        first sentence, so a response whose first sentence matched the reference
        scored a perfect 1.0 regardless of later sentences.
        """
        metric = BleuScore()

        reference = "The cat sat on the mat. It was a sunny day."
        # First sentence matches exactly; the second differs entirely.
        response = "The cat sat on the mat. Dogs barked."

        result = await metric.ascore(reference=reference, response=response)

        # The buggy implementation returned 1.0 (only first sentence compared).
        assert result.value < 1.0
