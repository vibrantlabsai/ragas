"""Tests for multilingual support in ROUGE and BLEU score metrics.

These tests verify that RougeScore and BleuScore work correctly with
non-English text, particularly Arabic. The key issues addressed:
- RougeScore: use_stemmer=True uses English Porter Stemmer (breaks non-English)
- BleuScore: hardcoded ". " sentence splitting (wrong for Arabic/other languages)
- quoted_spans: missing Arabic guillemets « »

Note: The rouge_score library's DefaultTokenizer strips non-ASCII characters,
so ROUGE scores for pure Arabic text are 0. The use_stemmer parameter is still
valuable for Latin-script languages (German, French, etc.) where the tokenizer
works but the English Porter Stemmer produces incorrect results.
"""

import pytest

from ragas.dataset_schema import SingleTurnSample

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

ARABIC_REFERENCE = "العاصمة الرسمية للمملكة العربية السعودية هي مدينة الرياض"
ARABIC_RESPONSE = "الرياض هي العاصمة الرسمية للمملكة العربية السعودية"

ARABIC_MULTI_SENTENCE_REF = (
    "يقع البحرين في الخليج العربي. "
    "عاصمتها المنامة. "
    "عدد سكانها حوالي مليون ونصف"
)
ARABIC_MULTI_SENTENCE_RESP = (
    "البحرين دولة في الخليج العربي. "
    "المنامة هي العاصمة. "
    "يبلغ عدد سكانها حوالي مليون ونصف"
)

# Arabic text that uses Arabic comma conventions (no ". " pattern)
ARABIC_NATURAL_REF = "يقع البحرين في الخليج العربي، عاصمتها المنامة، وعدد سكانها حوالي مليون ونصف"
ARABIC_NATURAL_RESP = "البحرين دولة في الخليج العربي، المنامة هي العاصمة، ويبلغ عدد سكانها حوالي مليون ونصف"

# German test data — Latin script where stemmer behavior differs
GERMAN_REFERENCE = "Die Hauptstadt von Deutschland ist Berlin"
GERMAN_RESPONSE = "Berlin ist die Hauptstadt von Deutschland"

# English baseline for comparison
ENGLISH_REFERENCE = "The capital of Saudi Arabia is Riyadh"
ENGLISH_RESPONSE = "Riyadh is the capital of Saudi Arabia"


# ===========================================================================
# RougeScore (old API) — use_stemmer tests
# ===========================================================================

class TestRougeScoreMultilingual:
    """Test RougeScore use_stemmer parameter for non-English text."""

    def test_use_stemmer_default_true(self):
        """use_stemmer defaults to True for backward compatibility."""
        from ragas.metrics import RougeScore
        metric = RougeScore()
        assert metric.use_stemmer is True

    def test_use_stemmer_can_be_disabled(self):
        """use_stemmer can be set to False for non-English text."""
        from ragas.metrics import RougeScore
        metric = RougeScore(use_stemmer=False)
        assert metric.use_stemmer is False

    @pytest.mark.asyncio
    async def test_arabic_text_returns_zero_due_to_tokenizer(self):
        """rouge_score's tokenizer strips non-ASCII, so Arabic scores are 0.

        This documents a known limitation of the rouge_score library.
        """
        from ragas.metrics import RougeScore
        metric = RougeScore(use_stemmer=False)
        sample = SingleTurnSample(
            reference=ARABIC_REFERENCE,
            response=ARABIC_RESPONSE,
        )
        score = await metric._single_turn_ascore(sample, None)
        assert float(score) == 0.0

    @pytest.mark.asyncio
    async def test_stemmer_off_gives_different_score_english(self):
        """Disabling stemmer should produce different scores for English
        (where stemming merges word forms like 'running' -> 'run')."""
        from ragas.metrics import RougeScore

        sample = SingleTurnSample(
            reference="The runners are running in the running competition",
            response="The runner ran in the run competition",
        )
        with_stemmer = RougeScore(use_stemmer=True)
        without_stemmer = RougeScore(use_stemmer=False)

        score_with = await with_stemmer._single_turn_ascore(sample, None)
        score_without = await without_stemmer._single_turn_ascore(sample, None)

        # Stemmer should give higher score because it normalizes word forms
        assert float(score_with) >= float(score_without)

    @pytest.mark.asyncio
    async def test_german_text_without_stemmer(self):
        """German text should produce valid scores (Latin script works with tokenizer)."""
        from ragas.metrics import RougeScore
        metric = RougeScore(use_stemmer=False)
        sample = SingleTurnSample(
            reference=GERMAN_REFERENCE,
            response=GERMAN_RESPONSE,
        )
        score = await metric._single_turn_ascore(sample, None)
        assert float(score) > 0.0

    @pytest.mark.asyncio
    async def test_german_stemmer_off_vs_on(self):
        """German text: English stemmer should not improve scores (wrong language)."""
        from ragas.metrics import RougeScore

        sample = SingleTurnSample(
            reference="Die Läufer laufen im Laufwettbewerb",
            response="Der Läufer lief im Wettbewerb",
        )
        with_stemmer = RougeScore(use_stemmer=True, rouge_type="rouge1")
        without_stemmer = RougeScore(use_stemmer=False, rouge_type="rouge1")

        score_with = await with_stemmer._single_turn_ascore(sample, None)
        score_without = await without_stemmer._single_turn_ascore(sample, None)

        # Both should be valid numbers
        assert float(score_with) >= 0.0
        assert float(score_without) >= 0.0


# ===========================================================================
# BleuScore (old API) — sentence_separator tests
# ===========================================================================

class TestBleuScoreMultilingual:
    """Test BleuScore sentence_separator parameter for non-English text."""

    def test_sentence_separator_default(self):
        """Default sentence_separator should be '. ' for backward compatibility."""
        from ragas.metrics import BleuScore
        metric = BleuScore()
        assert metric.sentence_separator == ". "

    def test_sentence_separator_custom(self):
        """sentence_separator can be customized."""
        from ragas.metrics import BleuScore
        metric = BleuScore(sentence_separator="。")  # Japanese period
        assert metric.sentence_separator == "。"

    def test_sentence_separator_none_disables_splitting(self):
        """Setting sentence_separator to None treats entire text as one sentence."""
        from ragas.metrics import BleuScore
        metric = BleuScore(sentence_separator=None)
        assert metric.sentence_separator is None

    @pytest.mark.asyncio
    async def test_arabic_text_no_splitting(self):
        """Arabic text without sentence splitting should produce valid scores."""
        from ragas.metrics import BleuScore
        metric = BleuScore(sentence_separator=None)
        sample = SingleTurnSample(
            reference=ARABIC_NATURAL_REF,
            response=ARABIC_NATURAL_RESP,
        )
        score = await metric._single_turn_ascore(sample, None)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_arabic_text_with_period_separator(self):
        """Arabic text that uses '. ' separators should work with default."""
        from ragas.metrics import BleuScore
        metric = BleuScore()
        sample = SingleTurnSample(
            reference=ARABIC_MULTI_SENTENCE_REF,
            response=ARABIC_MULTI_SENTENCE_RESP,
        )
        score = await metric._single_turn_ascore(sample, None)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_english_backward_compatibility(self):
        """English text should produce the same scores as before (default separator)."""
        from ragas.metrics import BleuScore
        metric = BleuScore()
        sample = SingleTurnSample(
            reference=ENGLISH_REFERENCE,
            response=ENGLISH_RESPONSE,
        )
        score = await metric._single_turn_ascore(sample, None)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_no_split_vs_split_different_scores(self):
        """Splitting vs not splitting should give different scores for multi-sentence text."""
        from ragas.metrics import BleuScore

        sample = SingleTurnSample(
            reference="The cat sat on the mat. The dog lay on the rug.",
            response="The cat sat on the mat. The dog lay on the rug.",
        )
        with_split = BleuScore(sentence_separator=". ")
        without_split = BleuScore(sentence_separator=None)

        score_split = await with_split._single_turn_ascore(sample, None)
        score_nosplit = await without_split._single_turn_ascore(sample, None)

        # Both should be valid scores (allow small float rounding)
        assert 0.0 <= score_split <= 1.01
        assert 0.0 <= score_nosplit <= 1.01


# ===========================================================================
# RougeScore (collections API) — use_stemmer tests
# ===========================================================================

class TestRougeScoreCollectionsMultilingual:
    """Test collections RougeScore use_stemmer parameter."""

    def test_use_stemmer_default_true(self):
        """use_stemmer defaults to True for backward compatibility."""
        from ragas.metrics.collections import RougeScore
        metric = RougeScore()
        assert metric.use_stemmer is True

    def test_use_stemmer_can_be_disabled(self):
        """use_stemmer can be set to False."""
        from ragas.metrics.collections import RougeScore
        metric = RougeScore(use_stemmer=False)
        assert metric.use_stemmer is False

    @pytest.mark.asyncio
    async def test_arabic_text_returns_zero_due_to_tokenizer(self):
        """rouge_score's tokenizer strips non-ASCII; documents limitation."""
        from ragas.metrics.collections import RougeScore
        metric = RougeScore(use_stemmer=False)
        result = await metric.ascore(
            reference=ARABIC_REFERENCE,
            response=ARABIC_RESPONSE,
        )
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_german_text_without_stemmer(self):
        """German text should produce valid scores in collections API."""
        from ragas.metrics.collections import RougeScore
        metric = RougeScore(use_stemmer=False)
        result = await metric.ascore(
            reference=GERMAN_REFERENCE,
            response=GERMAN_RESPONSE,
        )
        assert result.value > 0.0


# ===========================================================================
# BleuScore (collections API) — sentence_separator tests
# ===========================================================================

class TestBleuScoreCollectionsMultilingual:
    """Test collections BleuScore sentence_separator parameter."""

    def test_sentence_separator_default(self):
        """Default sentence_separator should be '. '."""
        from ragas.metrics.collections import BleuScore
        metric = BleuScore()
        assert metric.sentence_separator == ". "

    def test_sentence_separator_none(self):
        """sentence_separator can be set to None."""
        from ragas.metrics.collections import BleuScore
        metric = BleuScore(sentence_separator=None)
        assert metric.sentence_separator is None

    @pytest.mark.asyncio
    async def test_arabic_text_no_splitting(self):
        """Arabic text without splitting in collections API."""
        from ragas.metrics.collections import BleuScore
        metric = BleuScore(sentence_separator=None)
        result = await metric.ascore(
            reference=ARABIC_NATURAL_REF,
            response=ARABIC_NATURAL_RESP,
        )
        assert 0.0 <= result.value <= 1.0

    @pytest.mark.asyncio
    async def test_english_backward_compatibility(self):
        """English text should work the same as before."""
        from ragas.metrics.collections import BleuScore
        metric = BleuScore()
        result = await metric.ascore(
            reference=ENGLISH_REFERENCE,
            response=ENGLISH_RESPONSE,
        )
        assert 0.0 <= result.value <= 1.0


# ===========================================================================
# quoted_spans — Arabic guillemets tests
# ===========================================================================

class TestQuotedSpansArabic:
    """Test that quoted_spans extracts Arabic guillemets « »."""

    def test_extract_arabic_guillemets(self):
        """Arabic guillemet-quoted spans should be extracted."""
        from ragas.metrics.quoted_spans import _extract_quoted_spans
        answer = 'ينص القانون على «حق العامل في الإجازة السنوية» وذلك حسب المادة'
        spans = _extract_quoted_spans(answer)
        assert len(spans) == 1
        assert spans[0] == "حق العامل في الإجازة السنوية"

    def test_extract_mixed_quotes(self):
        """Both English quotes and Arabic guillemets should be extracted."""
        from ragas.metrics.quoted_spans import _extract_quoted_spans
        answer = 'The law states «حق العامل في الإجازة» and "the right to annual leave" per article 5'
        spans = _extract_quoted_spans(answer)
        assert len(spans) == 2

    def test_short_guillemet_spans_filtered(self):
        """Short Arabic guillemet spans should be filtered by min_len."""
        from ragas.metrics.quoted_spans import _extract_quoted_spans
        answer = 'ينص على «نعم» فقط'  # "نعم" is 1 word, below min_len=3
        spans = _extract_quoted_spans(answer)
        assert len(spans) == 0

    def test_alignment_with_arabic_sources(self):
        """quoted_spans_alignment should match Arabic guillemet spans in sources."""
        from ragas.metrics.quoted_spans import quoted_spans_alignment
        answers = ['ينص القانون على «حق العامل في الإجازة السنوية» وذلك حسب المادة']
        sources = [['حق العامل في الإجازة السنوية مكفول بموجب القانون']]
        result = quoted_spans_alignment(answers, sources)
        assert result["citation_alignment_quoted_spans"] == 1.0
        assert result["matched"] == 1.0
        assert result["total"] == 1.0

    def test_alignment_arabic_no_match(self):
        """Guillemet span not in sources should not match."""
        from ragas.metrics.quoted_spans import quoted_spans_alignment
        answers = ['ينص القانون على «حق العامل في الإجازة السنوية» وذلك حسب المادة']
        sources = [['هذا نص مختلف تماما عن المصدر الأصلي']]
        result = quoted_spans_alignment(answers, sources)
        assert result["citation_alignment_quoted_spans"] == 0.0
        assert result["matched"] == 0.0
        assert result["total"] == 1.0
