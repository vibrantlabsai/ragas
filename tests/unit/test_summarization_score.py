from ragas.metrics._summarization import SummarizationScore

def test_summarization_qa_score_empty_answers():
    metric = SummarizationScore()
    score = metric._compute_qa_score([])
    assert score == 0.0
