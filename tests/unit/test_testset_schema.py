from typing import Optional

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


@pytest.mark.parametrize("eval_sample", samples)
def test_testset_to_evaluation_dataset(eval_sample):
    testset_sample = RagasTestsetSample(
        eval_sample=eval_sample, synthesizer_name="test"
    )
    testset = RagasTestset(samples=[testset_sample, testset_sample])
    evaluation_dataset = testset.to_evaluation_dataset()
    assert evaluation_dataset == EvaluationDataset(samples=[eval_sample, eval_sample])


@pytest.mark.parametrize("user_input", ["What is X?", None])
def test_testset_from_dict_preserves_synthesizer_name(
    user_input: Optional[str],
) -> None:
    class CustomTestset(RagasTestset):
        pass

    row = {
        "reference": "Y",
        "synthesizer_name": "example",
    }
    if user_input is not None:
        row["user_input"] = user_input
    columns = {name: [value] for name, value in row.items()}
    testset = CustomTestset.from_dict(columns)
    assert isinstance(testset, CustomTestset)
    assert testset.samples == [
        RagasTestsetSample(
            eval_sample=SingleTurnSample(user_input=user_input, reference="Y"),
            synthesizer_name="example",
        )
    ]
    assert CustomTestset.from_dict([row]).samples == testset.samples
    assert row["synthesizer_name"] == "example"


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
