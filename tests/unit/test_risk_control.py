import pytest
from datasets import Dataset
from ragas.metrics._risk_control import risk_control_suite

# Sample dataset for testing
@pytest.fixture
def sample_dataset():
    return Dataset.from_list([
        # 2 AK cases (True Positives)
        {"ground_truth_answerable": True, "model_decision": "kept"},
        {"ground_truth_answerable": True, "model_decision": "kept"},
        # 1 UK case (False Positive / Risky)
        {"ground_truth_answerable": False, "model_decision": "kept"},
        # 3 UD cases (True Negatives)
        {"ground_truth_answerable": False, "model_decision": "discarded"},
        {"ground_truth_answerable": False, "model_decision": "discarded"},
        {"ground_truth_answerable": False, "model_decision": "discarded"},
        # 2 AD cases (False Negatives / Missed Opportunity)
        {"ground_truth_answerable": True, "model_decision": "discarded"},
        {"ground_truth_answerable": True, "model_decision": "discarded"},
    ])

def test_risk_control_suite_calculations(sample_dataset):
    """
    Tests the core calculations based on the sample dataset.
    Counts: AK=2, UK=1, UD=3, AD=2
    Total Kept = 3, Total Unanswerable = 4, Total Decisions = 8
    """
    risk_metrics = risk_control_suite(sample_dataset)
    scores = risk_metrics[0].calculator.get_scores() # All metrics share the calculator

    # Expected Risk = UK / (AK + UK) = 1 / 3 = 0.333...
    assert scores["risk"] == pytest.approx(1/3)
    # Expected Carefulness = UD / (UK + UD) = 3 / 4 = 0.75
    assert scores["carefulness"] == 0.75
    # Expected Alignment = (AK + UD) / Total = (2 + 3) / 8 = 0.625
    assert scores["alignment"] == 0.625
    # Expected Coverage = (AK + UK) / Total = 3 / 8 = 0.375
    assert scores["coverage"] == 0.375

def test_edge_case_no_kept_answers():
    dataset = Dataset.from_list([
        {"ground_truth_answerable": False, "model_decision": "discarded"},
        {"ground_truth_answerable": True, "model_decision": "discarded"},
    ])
    risk_metrics = risk_control_suite(dataset)
    scores = risk_metrics[0].calculator.get_scores()
    
    # Risk should be 0 if no answers are kept
    assert scores["risk"] == 0.0
    assert scores["coverage"] == 0.0

def test_edge_case_no_unanswerable_questions():
    dataset = Dataset.from_list([
        {"ground_truth_answerable": True, "model_decision": "kept"},
        {"ground_truth_answerable": True, "model_decision": "discarded"},
    ])
    risk_metrics = risk_control_suite(dataset)
    scores = risk_metrics[0].calculator.get_scores()

    # Carefulness should be 0 if there are no unanswerable questions to check
    assert scores["carefulness"] == 0.0

def test_missing_column_error():
    dataset = Dataset.from_list([{"model_decision": "kept"}]) # Missing ground_truth_answerable
    risk_metrics = risk_control_suite(dataset)
    with pytest.raises(ValueError, match="Missing required column 'ground_truth_answerable'"):
        risk_metrics[0].calculator.get_scores()