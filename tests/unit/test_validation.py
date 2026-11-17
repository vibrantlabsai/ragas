import typing as t
from dataclasses import dataclass, field

import pytest
from datasets import Dataset

from ragas.metrics.base import MetricType
from ragas.validation import (
    remap_column_names,
    validate_required_columns,
    validate_supported_metrics,
)

column_maps = [
    {
        "question": "query",
        "answer": "rag_answer",
        "contexts": "rag_contexts",
        "ground_truth": "original_answer",
    },  # all columns present
    {
        "question": "query",
        "answer": "rag_answer",
    },  # subset of columns present
]


@pytest.mark.parametrize(
    "response", [None, "An answer"], ids=["missing-response", "complete-columns"]
)
def test_validate_required_columns(response: t.Optional[str]):
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics.base import SingleTurnMetric

    @dataclass
    class MockMetric(SingleTurnMetric):
        name: str = "mock_metric"
        _required_columns: t.Dict[MetricType, t.Set[str]] = field(
            default_factory=lambda: {MetricType.SINGLE_TURN: {"user_input", "response"}}
        )

        def init(self, run_config):
            pass

        async def _single_turn_ascore(self, sample, callbacks):
            return 0.0

    m = MockMetric()
    sample1 = SingleTurnSample(user_input="What is X", response=response)
    sample2 = SingleTurnSample(user_input="What is Z", response=response)
    ds = EvaluationDataset(samples=[sample1, sample2])
    if response is None:
        with pytest.raises(
            ValueError, match=r"\[mock_metric\].*additional columns.*response"
        ):
            validate_required_columns(ds, [m])
    else:
        validate_required_columns(ds, [m])


def test_valid_data_type():
    from ragas.dataset_schema import EvaluationDataset, MultiTurnSample
    from ragas.messages import HumanMessage
    from ragas.metrics.base import MetricWithLLM, SingleTurnMetric

    @dataclass
    class MockMetric(MetricWithLLM, SingleTurnMetric):
        name = "mock_metric"
        _required_columns: t.Dict[MetricType, t.Set[str]] = field(
            default_factory=lambda: {MetricType.SINGLE_TURN: {"user_input"}}
        )

        def init(self, run_config):
            pass

        async def _single_turn_ascore(self, sample, callbacks):
            return 0.0

        async def _ascore(self, row, callbacks):
            return 0.0

    m = MockMetric()
    sample1 = MultiTurnSample(user_input=[HumanMessage(content="What is X")])
    sample2 = MultiTurnSample(user_input=[HumanMessage(content="What is X")])
    ds = EvaluationDataset(samples=[sample1, sample2])
    with pytest.raises(ValueError):
        validate_supported_metrics(ds, [m])


@pytest.mark.parametrize("column_map", column_maps)
def test_column_remap(column_map):
    """
    test cases:
    - extra columns present in the dataset
    - not all columsn selected
    - column names are different
    """
    TEST_DATASET = Dataset.from_dict(
        {
            "query": [""],
            "rag_answer": [""],
            "rag_contexts": [[""]],
            "original_answer": [""],
            "another_column": [""],
            "rag_answer_v2": [""],
            "rag_contexts_v2": [[""]],
        }
    )
    remapped_dataset = remap_column_names(TEST_DATASET, column_map)
    assert all(col in remapped_dataset.column_names for col in column_map.keys())


def test_column_remap_omit():
    TEST_DATASET = Dataset.from_dict(
        {
            "query": [""],
            "answer": [""],
            "contexts": [[""]],
        }
    )

    column_map = {
        "question": "query",
        "contexts": "contexts",
        "answer": "answer",
    }

    remapped_dataset = remap_column_names(TEST_DATASET, column_map)
    assert remapped_dataset.column_names == ["question", "answer", "contexts"]
