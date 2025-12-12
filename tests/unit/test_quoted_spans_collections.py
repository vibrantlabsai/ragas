"""Tests for QuotedSpansAlignment metric (collections implementation)."""

import pytest

from ragas.metrics.collections import QuotedSpansAlignment
from ragas.metrics.collections.quoted_spans.util import (
    count_matched_spans,
    extract_quoted_spans,
    normalize_text,
)


class TestQuotedSpansUtilities:
    """Test cases for utility functions."""

    def test_normalize_text_basic(self):
        """Test basic text normalization."""
        assert normalize_text("  Hello   World  ") == "hello world"

    def test_normalize_text_multiline(self):
        """Test normalization of multiline text."""
        assert normalize_text("hello\n\nworld") == "hello world"

    def test_extract_quoted_spans_double_quotes(self):
        """Test extraction with double quotes."""
        text = (
            'The study found that "machine learning improves accuracy" in most cases.'
        )
        spans = extract_quoted_spans(text, min_len=3)
        assert spans == ["machine learning improves accuracy"]

    def test_extract_quoted_spans_single_quotes(self):
        """Test extraction with single quotes."""
        text = "He said 'the results are significant' and we agreed."
        spans = extract_quoted_spans(text, min_len=3)
        assert spans == ["the results are significant"]

    def test_extract_quoted_spans_curly_quotes(self):
        """Test extraction with curly/smart quotes."""
        text = (
            "The paper states \u201cdeep learning outperforms baselines\u201d clearly."
        )
        spans = extract_quoted_spans(text, min_len=3)
        assert spans == ["deep learning outperforms baselines"]

    def test_extract_quoted_spans_min_len_filter(self):
        """Test that short spans are filtered out."""
        text = '"short" and "this is a longer quoted span"'
        spans = extract_quoted_spans(text, min_len=3)
        assert spans == ["this is a longer quoted span"]
        assert "short" not in spans

    def test_extract_quoted_spans_empty(self):
        """Test extraction with no quotes."""
        text = "No quotes in this text at all."
        spans = extract_quoted_spans(text, min_len=3)
        assert spans == []

    def test_extract_quoted_spans_multiple(self):
        """Test extraction of multiple quoted spans."""
        text = '"first span here" and then "second span here" in text'
        spans = extract_quoted_spans(text, min_len=3)
        assert len(spans) == 2
        assert "first span here" in spans
        assert "second span here" in spans

    def test_count_matched_spans_all_match(self):
        """Test when all spans are found in sources."""
        spans = ["machine learning", "deep learning models"]
        sources = ["Machine learning and deep learning models are popular."]
        matched, total = count_matched_spans(spans, sources, casefold=True)
        assert matched == 2
        assert total == 2

    def test_count_matched_spans_none_match(self):
        """Test when no spans are found in sources."""
        spans = ["quantum computing", "neural networks"]
        sources = ["This is about cooking recipes and gardening tips."]
        matched, total = count_matched_spans(spans, sources, casefold=True)
        assert matched == 0
        assert total == 2

    def test_count_matched_spans_partial_match(self):
        """Test when some spans match."""
        spans = ["machine learning", "quantum physics"]
        sources = ["Machine learning is powerful."]
        matched, total = count_matched_spans(spans, sources, casefold=True)
        assert matched == 1
        assert total == 2

    def test_count_matched_spans_case_sensitive(self):
        """Test case-sensitive matching."""
        spans = ["Machine Learning"]
        sources = ["machine learning is great"]
        matched, total = count_matched_spans(spans, sources, casefold=False)
        assert matched == 0
        assert total == 1

    def test_count_matched_spans_empty_spans(self):
        """Test with empty spans list."""
        matched, total = count_matched_spans([], ["some source"], casefold=True)
        assert matched == 0
        assert total == 0


class TestQuotedSpansAlignmentCollections:
    """Test cases for QuotedSpansAlignment metric from collections."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        metric = QuotedSpansAlignment()
        assert metric.name == "quoted_spans_alignment"
        assert metric.casefold is True
        assert metric.min_span_words == 3

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        metric = QuotedSpansAlignment(
            name="custom_metric", casefold=False, min_span_words=5
        )
        assert metric.name == "custom_metric"
        assert metric.casefold is False
        assert metric.min_span_words == 5

    @pytest.mark.asyncio
    async def test_perfect_alignment(self):
        """Test when all quoted spans are found in sources."""
        metric = QuotedSpansAlignment()

        response = 'The study shows "machine learning improves results" significantly.'
        sources = ["Machine learning improves results in many domains."]

        result = await metric.ascore(response=response, retrieved_contexts=sources)
        assert result.value == 1.0
        assert "1/1" in result.reason

    @pytest.mark.asyncio
    async def test_no_alignment(self):
        """Test when no quoted spans are found in sources."""
        metric = QuotedSpansAlignment()

        response = (
            'According to the paper, "quantum entanglement enables teleportation".'
        )
        sources = ["This document discusses cooking and gardening."]

        result = await metric.ascore(response=response, retrieved_contexts=sources)
        assert result.value == 0.0
        assert "0/1" in result.reason

    @pytest.mark.asyncio
    async def test_partial_alignment(self):
        """Test partial match scenario."""
        metric = QuotedSpansAlignment()

        response = '"Machine learning is powerful" and "quantum physics is complex".'
        sources = ["Machine learning is powerful and useful."]

        result = await metric.ascore(response=response, retrieved_contexts=sources)
        assert result.value == 0.5
        assert "1/2" in result.reason

    @pytest.mark.asyncio
    async def test_no_quotes_in_response(self):
        """Test when response has no quoted spans."""
        metric = QuotedSpansAlignment()

        response = "This response has no quoted spans at all."
        sources = ["Some source text here."]

        result = await metric.ascore(response=response, retrieved_contexts=sources)
        assert result.value == 1.0
        assert "No quoted spans found" in result.reason

    @pytest.mark.asyncio
    async def test_multiple_sources(self):
        """Test with multiple source documents."""
        metric = QuotedSpansAlignment()

        response = 'The paper states "deep learning outperforms baselines".'
        sources = [
            "First document about cooking.",
            "Deep learning outperforms baselines in many tasks.",
            "Third document about sports.",
        ]

        result = await metric.ascore(response=response, retrieved_contexts=sources)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self):
        """Test case-insensitive matching (default)."""
        metric = QuotedSpansAlignment(casefold=True)

        response = 'The report says "MACHINE LEARNING IS POWERFUL".'
        sources = ["machine learning is powerful and useful."]

        result = await metric.ascore(response=response, retrieved_contexts=sources)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_case_sensitive_matching(self):
        """Test case-sensitive matching."""
        metric = QuotedSpansAlignment(casefold=False)

        response = 'The report says "MACHINE LEARNING IS POWERFUL".'
        sources = ["machine learning is powerful and useful."]

        result = await metric.ascore(response=response, retrieved_contexts=sources)
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_min_span_words_filter(self):
        """Test minimum span words filter."""
        metric = QuotedSpansAlignment(min_span_words=5)

        response = '"short span" and "this is a much longer quoted span here".'
        sources = ["This is a much longer quoted span here for testing."]

        result = await metric.ascore(response=response, retrieved_contexts=sources)
        assert result.value == 1.0
        assert "1/1" in result.reason

    @pytest.mark.asyncio
    async def test_invalid_response_type(self):
        """Test with invalid response type."""
        metric = QuotedSpansAlignment()

        result = await metric.ascore(response=123, retrieved_contexts=["text"])
        assert result.value == 0.0
        assert "Invalid input" in result.reason

    @pytest.mark.asyncio
    async def test_invalid_contexts_type(self):
        """Test with invalid contexts type."""
        metric = QuotedSpansAlignment()

        result = await metric.ascore(
            response="some text", retrieved_contexts="not a list"
        )
        assert result.value == 0.0
        assert "Invalid input" in result.reason

    @pytest.mark.asyncio
    async def test_empty_contexts(self):
        """Test with empty contexts list."""
        metric = QuotedSpansAlignment()

        response = 'The study found "important results here".'
        result = await metric.ascore(response=response, retrieved_contexts=[])
        assert result.value == 0.0
        assert "0/1" in result.reason

    @pytest.mark.asyncio
    async def test_whitespace_normalization(self):
        """Test that whitespace is normalized in matching."""
        metric = QuotedSpansAlignment()

        response = 'The paper says "machine   learning    improves  results".'
        sources = ["Machine learning improves results significantly."]

        result = await metric.ascore(response=response, retrieved_contexts=sources)
        assert result.value == 1.0

    def test_sync_score_method(self):
        """Test synchronous score method."""
        metric = QuotedSpansAlignment()

        response = 'The study shows "machine learning improves results".'
        sources = ["Machine learning improves results in many domains."]

        result = metric.score(response=response, retrieved_contexts=sources)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_curly_quotes(self):
        """Test with curly/smart quotes."""
        metric = QuotedSpansAlignment()

        response = "The document states \u201cneural networks are effective\u201d for classification."
        sources = ["Neural networks are effective for many tasks."]

        result = await metric.ascore(response=response, retrieved_contexts=sources)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_backtick_quotes(self):
        """Test with backtick quotes."""
        metric = QuotedSpansAlignment()

        response = "The code says `return the final result` at the end."
        sources = ["return the final result"]

        result = await metric.ascore(response=response, retrieved_contexts=sources)
        assert result.value == 1.0
