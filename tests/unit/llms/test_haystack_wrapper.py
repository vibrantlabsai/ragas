"""Tests for the Haystack LLM wrapper."""

import sys
import types

import pytest
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import StringPromptValue

from ragas.llms.haystack_wrapper import HaystackLLMWrapper


class FakeHaystackGenerator:
    """Haystack generator component: run() returns replies at the top level."""

    def __init__(self, reply="hello world"):
        self.reply = reply
        self.prompts = []

    def run(self, prompt, generation_kwargs=None):
        self.prompts.append(prompt)
        return {"replies": [self.reply], "meta": [{"model": "fake-model"}]}


class FakeAsyncPipeline:
    """Haystack pipeline: component output is nested under the component name."""

    def __init__(self):
        self.components = {}

    def add_component(self, name, component):
        self.components[name] = component

    async def run_async(self, data):
        ((name, inputs),) = data.items()
        return {name: self.components[name].run(**inputs)}


class TestHaystackLLMWrapper:
    @pytest.fixture
    def fake_haystack(self, monkeypatch):
        """Install the haystack-ai surface the wrapper is written against.

        The generator modules the wrapper imports were removed in
        haystack-ai 3.x, so they are faked here to keep the tests
        independent of the installed haystack-ai version.
        """
        haystack = types.ModuleType("haystack")
        haystack.AsyncPipeline = FakeAsyncPipeline
        monkeypatch.setitem(sys.modules, "haystack", haystack)

        generator_classes = {
            "haystack.components.generators.azure": "AzureOpenAIGenerator",
            "haystack.components.generators.hugging_face_api": "HuggingFaceAPIGenerator",
            "haystack.components.generators.hugging_face_local": "HuggingFaceLocalGenerator",
            "haystack.components.generators.openai": "OpenAIGenerator",
        }
        for module_name, class_name in generator_classes.items():
            module = types.ModuleType(module_name)
            setattr(module, class_name, FakeHaystackGenerator)
            monkeypatch.setitem(sys.modules, module_name, module)

    def test_generate_text_returns_component_reply(self, fake_haystack):
        """generate_text reads replies from the component's own output."""
        generator = FakeHaystackGenerator(reply="hello world")
        wrapper = HaystackLLMWrapper(haystack_generator=generator)

        result = wrapper.generate_text(StringPromptValue(text="hi"))

        assert isinstance(result, LLMResult)
        assert result.generations[0][0].text == "hello world"
        assert generator.prompts == ["hi"]

    @pytest.mark.asyncio
    async def test_agenerate_text_returns_reply(self, fake_haystack):
        """agenerate_text reads replies from the pipeline output."""
        generator = FakeHaystackGenerator(reply="hello world")
        wrapper = HaystackLLMWrapper(haystack_generator=generator)

        result = await wrapper.agenerate_text(StringPromptValue(text="hi"))

        assert isinstance(result, LLMResult)
        assert result.generations[0][0].text == "hello world"
