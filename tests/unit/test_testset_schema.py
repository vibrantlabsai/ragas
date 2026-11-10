from copy import deepcopy

import pytest

from ragas.dataset_schema import (
    EvaluationDataset,
    HumanMessage,
    MultiTurnSample,
    SingleTurnSample,
)
from ragas.testset.synthesizers.testset_schema import (
    Testset as RagasTestset,
    TestsetSample as RagasTestsetSample,
)

samples = [
    SingleTurnSample(user_input="What is X", response="Y"),
    MultiTurnSample(
        user_input=[HumanMessage(content="What is X")],
        reference="Y",
    ),
]


@pytest.fixture(params=samples, ids=["single_turn", "multi_turn"])
def testset_input(request):
    sample = request.param.model_copy(deep=True)
    sample.rubrics = {"1": "Incorrect", "5": "Correct"}
    if isinstance(sample, SingleTurnSample):
        sample.retrieved_contexts = ["First context", "Second context"]
    else:
        sample.user_input[0].metadata = {"source": {"tags": ["first", "second"]}}
    testset = RagasTestset(
        samples=[RagasTestsetSample(eval_sample=sample, synthesizer_name="test")]
    )
    return testset, testset.to_list()


def test_testset_from_list_preserves_input(testset_input):
    expected, data = testset_input
    original = deepcopy(data)

    assert RagasTestset.from_list(data) == expected
    assert data == original


def test_testset_from_list_can_reuse_input(testset_input):
    expected, data = testset_input

    first = RagasTestset.from_list(data)
    second = RagasTestset.from_list(data)

    assert first == second == expected


def test_testset_from_list_accepts_repeated_row(testset_input):
    expected, data = testset_input
    row = data[0]

    testset = RagasTestset.from_list([row, row])

    assert testset.samples == expected.samples * 2
    assert testset.to_list() == data * 2


@pytest.mark.parametrize("eval_sample", samples)
def test_testset_to_evaluation_dataset(eval_sample):
    testset_sample = RagasTestsetSample(
        eval_sample=eval_sample, synthesizer_name="test"
    )
    testset = RagasTestset(samples=[testset_sample, testset_sample])
    evaluation_dataset = testset.to_evaluation_dataset()
    assert evaluation_dataset == EvaluationDataset(samples=[eval_sample, eval_sample])


@pytest.mark.parametrize("eval_sample", samples)
def test_testset_save_load_csv(tmpdir, eval_sample):
    testset_sample = RagasTestsetSample(
        eval_sample=eval_sample, synthesizer_name="test"
    )
    testset = RagasTestset(samples=[testset_sample, testset_sample])
    testset.to_csv(tmpdir / "csvfile.csv")


@pytest.mark.parametrize("eval_sample", samples)
def test_testset_save_load_jsonl(tmpdir, eval_sample):
    testset_sample = RagasTestsetSample(
        eval_sample=eval_sample, synthesizer_name="test"
    )
    testset = RagasTestset(samples=[testset_sample, testset_sample])
    testset.to_jsonl(tmpdir / "jsonlfile.jsonl")
    loaded_testset = RagasTestset.from_jsonl(tmpdir / "jsonlfile.jsonl")
    assert loaded_testset == testset


@pytest.mark.parametrize("eval_sample", samples)
def test_testset_save_load_hf(tmpdir, eval_sample):
    testset_sample = RagasTestsetSample(
        eval_sample=eval_sample, synthesizer_name="test"
    )
    testset = RagasTestset(samples=[testset_sample, testset_sample])
    hf_testset = testset.to_hf_dataset()
    loaded_testset = RagasTestset.from_hf_dataset(hf_testset)
    assert loaded_testset == testset
