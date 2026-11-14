"""Tests for SummarizationScore metric (legacy implementation)."""

from ragas.metrics._summarization import SummarizationScore


class TestSummarizationScoreQaScore:
    """``_compute_qa_score`` needs no LLM — it's pure arithmetic over the
    already-generated answers list."""

    def test_all_correct_answers_scores_one(self):
        metric = SummarizationScore()
        assert metric._compute_qa_score(["1", "1", "1"]) == 1.0

    def test_no_correct_answers_scores_zero(self):
        metric = SummarizationScore()
        assert metric._compute_qa_score(["0", "0"]) == 0.0

    def test_mixed_answers_computes_ratio(self):
        metric = SummarizationScore()
        assert metric._compute_qa_score(["1", "0", "1", "0"]) == 0.5

    def test_empty_answers_scores_zero_instead_of_raising(self):
        """Regression: keyphrase or question extraction can legitimately
        return an empty list (already handled by logging an error and
        continuing), which used to cascade into `answers == []` and a
        ZeroDivisionError here instead of a score."""
        metric = SummarizationScore()
        assert metric._compute_qa_score([]) == 0.0
