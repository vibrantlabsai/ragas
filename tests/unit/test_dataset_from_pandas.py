import numpy as np
import pandas as pd

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.testset.synthesizers.testset_schema import Testset, TestsetSample


def test_from_pandas_skips_missing_cells():
    # pandas fills empty cells (for example from pd.read_csv) with NaN
    df = pd.DataFrame(
        {
            "user_input": ["q1", "q2"],
            "response": ["a1", "a2"],
            "reference": ["r1", np.nan],
        }
    )

    dataset = EvaluationDataset.from_pandas(df)

    assert dataset.samples[0].reference == "r1"
    assert dataset.samples[1].reference is None
    assert dataset.samples[1].response == "a2"


def test_from_pandas_keeps_list_cells_and_skips_missing_lists():
    df = pd.DataFrame(
        {
            "user_input": ["q1", "q2"],
            "retrieved_contexts": [["c1", "c2"], np.nan],
        }
    )

    dataset = EvaluationDataset.from_pandas(df)

    assert dataset.samples[0].retrieved_contexts == ["c1", "c2"]
    assert dataset.samples[1].retrieved_contexts is None


def test_from_pandas_round_trips_samples_with_different_fields():
    # to_pandas() fills the fields a sample does not have with NaN
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(user_input="q1", response="a1"),
            SingleTurnSample(user_input="q2", reference="r2"),
        ]
    )

    restored = EvaluationDataset.from_pandas(dataset.to_pandas())

    assert restored.samples == dataset.samples


def test_testset_from_pandas_round_trips_samples_with_different_fields():
    testset = Testset(
        samples=[
            TestsetSample(
                eval_sample=SingleTurnSample(
                    user_input="q1", response="a1", reference="r1"
                ),
                synthesizer_name="s1",
            ),
            TestsetSample(
                eval_sample=SingleTurnSample(user_input="q2", reference="r2"),
                synthesizer_name="s2",
            ),
        ]
    )

    restored = Testset.from_pandas(testset.to_pandas())

    assert restored.samples == testset.samples
