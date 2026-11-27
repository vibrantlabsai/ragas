from unittest.mock import AsyncMock, Mock

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import generate


@pytest.mark.parametrize("failure_stage", ["scenario", "sample"])
@pytest.mark.parametrize("raise_exceptions", [False, True])
def test_generate_handles_failed_jobs(failure_stage, raise_exceptions):
    generator = generate.TestsetGenerator(
        llm=Mock(),
        embedding_model=Mock(),
        persona_list=[Persona(name="Reader", role_description="Reads documents")],
    )

    synthesizers = []
    for name in ("first", "broken", "last"):
        synth = Mock()
        synth.name = name
        synth.generate_scenarios = AsyncMock(return_value=[name])
        synth.generate_sample = AsyncMock(
            return_value=SingleTurnSample(user_input=name)
        )
        synthesizers.append(synth)

    failure = RuntimeError("synthetic generation failure")
    broken = synthesizers[1]
    if failure_stage == "scenario":
        broken.generate_scenarios.side_effect = failure
    else:
        broken.generate_sample.side_effect = failure

    kwargs = {
        "testset_size": 3,
        "query_distribution": [(synth, 1 / 3) for synth in synthesizers],
        "raise_exceptions": raise_exceptions,
    }
    if raise_exceptions:
        with pytest.raises(RuntimeError, match="synthetic generation failure"):
            generator.generate(**kwargs)
        return

    testset = generator.generate(**kwargs)
    assert [
        (sample.synthesizer_name, sample.eval_sample.user_input)
        for sample in testset.samples
    ] == [("first", "first"), ("last", "last")]
    if failure_stage == "scenario":
        broken.generate_sample.assert_not_awaited()
