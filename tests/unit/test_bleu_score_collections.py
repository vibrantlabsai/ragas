"""Tests for BleuScore metric (collections implementation)."""

import pytest

try:
    from sacrebleu import corpus_bleu  # noqa: F401
except ImportError:
    pytest.skip("sacrebleu not available", allow_module_level=True)

from ragas.metrics.collections import BleuScore


class TestBleuScoreCollections:
    """Test cases for BleuScore metric from collections."""

    @pytest.mark.asyncio
    async def test_identical_chinese_text_with_zh_tokenizer(self):
        """Test identical Chinese text with the Chinese tokenizer."""
        metric = BleuScore(kwargs={"tokenize": "zh"})
        text = "今天天气很好，我们一起去公园散步。"

        result = await metric.ascore(reference=text, response=text)

        assert result.value == pytest.approx(1.0)
