import copy
import typing as t

import numpy as np
import pytest
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import StringPromptValue
from pydantic import BaseModel

from ragas.llms.base import BaseRagasLLM
from ragas.prompt import StringIO, StringPrompt
from ragas.run_config import RunConfig


class EchoLLM(BaseRagasLLM):
    def generate_text(  # type: ignore
        self,
        prompt: StringPromptValue,
        *args,
        **kwargs,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=prompt.to_string())]])

    async def agenerate_text(  # type: ignore
        self,
        prompt: StringPromptValue,
        *args,
        **kwargs,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=prompt.to_string())]])

    def is_finished(self, response: LLMResult) -> bool:
        return True


@pytest.mark.asyncio
async def test_string_prompt():
    echo_llm = EchoLLM(run_config=RunConfig())
    prompt = StringPrompt()
    assert await prompt.generate(data="hello", llm=echo_llm) == "hello"
    assert prompt.name == "string_prompt"


expected_generate_output_signature = """\
Please return the output in the following JSON format based on the StringIO model:
{
    "text": "str"
}\
"""


def test_process_fields():
    from enum import Enum

    from pydantic import BaseModel

    from ragas.prompt import PydanticPrompt, StringIO

    class Categories(str, Enum):
        science = "science"
        commerce = "commerce"
        agriculture = "agriculture"
        economics = "economics"

    class InputModel(BaseModel):
        category: Categories

    class JokeGenerator(PydanticPrompt[InputModel, StringIO]):
        instruction = "Generate a joke in the category of {category}."
        output_model = StringIO

    p = JokeGenerator()
    _ = p._generate_output_signature()

    # assert expected_generate_output_signature == generation


@pytest.mark.asyncio
async def test_pydantic_prompt_io():
    from ragas.prompt import PydanticPrompt, StringIO

    class Prompt(PydanticPrompt[StringIO, StringIO]):
        instruction = ""
        input_model = StringIO
        output_model = StringIO

    p = Prompt()
    assert p.input_model == StringIO
    assert p.output_model == StringIO

    assert p._generate_examples() == ""


def test_pydantic_prompt_examples():
    from ragas.prompt import PydanticPrompt

    class Prompt(PydanticPrompt[StringIO, StringIO]):
        instruction = ""
        input_model = StringIO
        output_model = StringIO
        examples = [
            (StringIO(text="hello"), StringIO(text="hello")),
            (StringIO(text="world"), StringIO(text="world")),
        ]

    _ = Prompt()
    # assert p.generate_examples() == "hello -> hello\nworld -> world"


def test_prompt_hash():
    from ragas.prompt import PydanticPrompt, StringIO

    class Prompt(PydanticPrompt[StringIO, StringIO]):
        instruction = "You are a helpful assistant."
        input_model = StringIO
        output_model = StringIO

    p = Prompt()
    p_copy = Prompt()
    assert hash(p) == hash(p_copy)
    assert p == p_copy
    p.instruction = "You are a helpful assistant. And some more"
    assert hash(p) != hash(p_copy)
    assert p != p_copy


def test_prompt_hash_in_ragas(fake_llm):
    # check with a prompt inside ragas
    from ragas.testset.synthesizers.multi_hop import MultiHopAbstractQuerySynthesizer

    synthesizer = MultiHopAbstractQuerySynthesizer(llm=fake_llm)
    prompts = synthesizer.get_prompts()
    for prompt in prompts.values():
        assert hash(prompt) == hash(prompt)
        assert prompt == prompt

    # change instruction and check if hash changes
    for prompt in prompts.values():
        old_prompt = copy.deepcopy(prompt)
        prompt.instruction = "You are a helpful assistant."
        assert hash(prompt) != hash(old_prompt)
        assert prompt != old_prompt


def test_prompt_save_load(tmp_path):
    from ragas.prompt import PydanticPrompt, StringIO

    class Prompt(PydanticPrompt[StringIO, StringIO]):
        instruction = "You are a helpful assistant."
        input_model = StringIO
        output_model = StringIO
        examples = [
            (StringIO(text="hello"), StringIO(text="hello")),
            (StringIO(text="world"), StringIO(text="world")),
        ]

    p = Prompt()
    file_path = tmp_path / "test_prompt.json"
    p.save(file_path)
    p1 = Prompt.load(file_path)
    assert hash(p) == hash(p1)
    assert p == p1


def test_prompt_save_load_language(tmp_path):
    from ragas.prompt import PydanticPrompt, StringIO

    class Prompt(PydanticPrompt[StringIO, StringIO]):
        instruction = "You are a helpful assistant."
        language = "spanish"
        input_model = StringIO
        output_model = StringIO
        examples = [
            (StringIO(text="hello"), StringIO(text="hello")),
            (StringIO(text="world"), StringIO(text="world")),
        ]

    p_spanish = Prompt()
    file_path = tmp_path / "test_prompt_spanish.json"
    p_spanish.save(file_path)
    p_spanish_loaded = Prompt.load(file_path)
    assert hash(p_spanish) == hash(p_spanish_loaded)
    assert p_spanish == p_spanish_loaded


def test_save_existing_prompt(tmp_path):
    from ragas.testset.synthesizers.prompts import ThemesPersonasMatchingPrompt

    p = ThemesPersonasMatchingPrompt()
    file_path = tmp_path / "test_prompt.json"
    p.save(file_path)
    p2 = ThemesPersonasMatchingPrompt.load(file_path)
    assert p == p2


def test_prompt_class_attributes():
    """
    We are using class attributes to store the prompt instruction and examples.
    We want to make sure there is no relationship between the class attributes
    and instance.
    """
    from ragas.testset.synthesizers.prompts import ThemesPersonasMatchingPrompt

    p = ThemesPersonasMatchingPrompt()
    p_another_instance = ThemesPersonasMatchingPrompt()
    assert p.instruction == p_another_instance.instruction
    assert p.examples == p_another_instance.examples
    p.instruction = "You are a helpful assistant."
    p.examples = []
    assert p.instruction != p_another_instance.instruction
    assert p.examples != p_another_instance.examples


@pytest.mark.asyncio
async def test_prompt_parse_retry():
    from ragas.exceptions import RagasOutputParserException
    from ragas.prompt import PydanticPrompt, StringIO

    class OutputModel(BaseModel):
        example: str

    class Prompt(PydanticPrompt[StringIO, OutputModel]):
        instruction = ""
        input_model = StringIO
        output_model = OutputModel

    echo_llm = EchoLLM(run_config=RunConfig())
    prompt = Prompt()
    with pytest.raises(RagasOutputParserException):
        await prompt.generate(
            data=StringIO(text="this prompt will be echoed back as invalid JSON"),
            llm=echo_llm,
        )


@pytest.mark.asyncio
async def test_prompt_fix_output_uses_nonzero_temperature():
    """Regression guard for the low-temperature JSON retry death-spiral.

    When the first generation produces output that fails to parse, ragas
    re-prompts the LLM to fix it. That fix attempt must run at a
    non-deterministic temperature -- otherwise a deterministic model (e.g. any
    provider at temperature ~0) re-emits the identical malformed output and
    every retry is wasted. Here we assert the initial attempt stays
    deterministic while every fix retry uses ``FIX_OUTPUT_FORMAT_TEMPERATURE``.
    """
    from ragas.exceptions import RagasOutputParserException
    from ragas.prompt import PydanticPrompt, StringIO
    from ragas.prompt.pydantic_prompt import FIX_OUTPUT_FORMAT_TEMPERATURE

    recorded_temperatures: t.List[t.Optional[float]] = []

    class RecordingLLM(BaseRagasLLM):
        """Always echoes back invalid JSON, recording each call's temperature."""

        def generate_text(  # type: ignore
            self,
            prompt: StringPromptValue,
            *args,
            **kwargs,
        ) -> LLMResult:
            return LLMResult(generations=[[Generation(text="not valid json")]])

        async def agenerate_text(  # type: ignore
            self,
            prompt: StringPromptValue,
            n: int = 1,
            temperature: t.Optional[float] = None,
            stop: t.Optional[t.List[str]] = None,
            callbacks: t.Any = None,
        ) -> LLMResult:
            recorded_temperatures.append(temperature)
            return LLMResult(generations=[[Generation(text="not valid json")]])

        def is_finished(self, response: LLMResult) -> bool:
            return True

    class OutputModel(BaseModel):
        example: str

    class Prompt(PydanticPrompt[StringIO, OutputModel]):
        instruction = ""
        input_model = StringIO
        output_model = OutputModel

    llm = RecordingLLM(run_config=RunConfig())
    prompt = Prompt()

    with pytest.raises(RagasOutputParserException):
        await prompt.generate(data=StringIO(text="x"), llm=llm)

    # The initial generation is deterministic (default temperature for n=1).
    assert recorded_temperatures, "LLM was never called"
    assert recorded_temperatures[0] == 0.01
    # Every subsequent call is a fix-output retry and must be non-deterministic.
    assert len(recorded_temperatures) > 1, "no fix-output retry was attempted"
    assert all(
        temp == FIX_OUTPUT_FORMAT_TEMPERATURE for temp in recorded_temperatures[1:]
    )


def cosine_similarity(v1: t.List[float], v2: t.List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    v1_array = np.array(v1)
    v2_array = np.array(v2)
    return np.dot(v1_array, v2_array) / (
        np.linalg.norm(v1_array) * np.linalg.norm(v2_array)
    )


@pytest.mark.skip(reason="TODO: Implement embedding calculation")
def test_in_memory_example_store():
    from ragas.prompt import InMemoryExampleStore

    class FakeInputModel(BaseModel):
        text: str
        embedding: t.List[float]

    class FakeOutputModel(BaseModel):
        text: str

    from tests.conftest import EchoEmbedding

    store = InMemoryExampleStore(embeddings=EchoEmbedding())
    store.add_example(
        FakeInputModel(text="hello", embedding=[1, 2, 3]),
        FakeOutputModel(text="hello"),
    )
    store.add_example(
        FakeInputModel(text="world", embedding=[1, 2, 4]),
        FakeOutputModel(text="world"),
    )
    assert store.get_examples(FakeInputModel(text="hello", embedding=[1, 2, 3])) == [
        FakeOutputModel(text="hello")
    ]
