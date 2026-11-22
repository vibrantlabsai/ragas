from ragas.metrics._summarization import SummarizationScore


def test_length_penalty_disabled_returns_full_qa_score():
    # With no length penalty the score is the QA score alone, not qa_score * (1 - coeff).
    m = SummarizationScore(length_penalty=False)
    assert m._compute_score({"qa_score": 1.0}) == 1.0
    assert m._compute_score({"qa_score": 0.8}) == 0.8


def test_length_penalty_enabled_unchanged():
    m = SummarizationScore(length_penalty=True)
    assert m._compute_score({"qa_score": 1.0, "conciseness_score": 0.4}) == 0.7
