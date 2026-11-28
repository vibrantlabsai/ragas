import math

from ragas.metrics._summarization import SummarizationScore


def test_compute_qa_score_empty_answers_returns_nan():
    metric = SummarizationScore(llm=None)
    score = metric._compute_qa_score([])
    assert math.isnan(score)


def test_compute_qa_score_counts_correct_answers():
    metric = SummarizationScore(llm=None)
    assert metric._compute_qa_score(["1", "0"]) == 0.5
