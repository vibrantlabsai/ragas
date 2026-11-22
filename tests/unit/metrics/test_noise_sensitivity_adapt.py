"""Unit tests for NoiseSensitivity adapt_prompts functionality.

This module tests that NoiseSensitivity properly supports the adapt_prompts() method
for multilingual adaptation, which was broken due to the use of BasePrompt instead
of PydanticPrompt (issue #2569).
"""

import pytest

from ragas.metrics.collections.noise_sensitivity.util import (
    StatementFaithfulnessPrompt,
    StatementGeneratorPrompt,
)
from ragas.prompt import PydanticPrompt


class TestNoiseSensitivityPromptAdaptation:
    """Tests for NoiseSensitivity prompt adaptation support."""

    def test_statement_generator_prompt_is_pydantic_prompt(self):
        """StatementGeneratorPrompt should inherit from PydanticPrompt."""
        prompt = StatementGeneratorPrompt()
        assert isinstance(prompt, PydanticPrompt)

    def test_statement_faithfulness_prompt_is_pydantic_prompt(self):
        """StatementFaithfulnessPrompt should inherit from PydanticPrompt."""
        prompt = StatementFaithfulnessPrompt()
        assert isinstance(prompt, PydanticPrompt)

    def test_statement_generator_prompt_has_examples(self):
        """StatementGeneratorPrompt should have examples for adaptation."""
        prompt = StatementGeneratorPrompt()
        assert hasattr(prompt, "examples")
        assert len(prompt.examples) > 0
        # Check example structure
        example_input, example_output = prompt.examples[0]
        assert hasattr(example_input, "question")
        assert hasattr(example_input, "text")
        assert hasattr(example_output, "statements")

    def test_statement_faithfulness_prompt_has_examples(self):
        """StatementFaithfulnessPrompt should have examples for adaptation."""
        prompt = StatementFaithfulnessPrompt()
        assert hasattr(prompt, "examples")
        assert len(prompt.examples) > 0
        # Check example structure
        example_input, example_output = prompt.examples[0]
        assert hasattr(example_input, "context")
        assert hasattr(example_input, "statements")
        assert hasattr(example_output, "statements")

    def test_statement_generator_prompt_has_instruction(self):
        """StatementGeneratorPrompt should have an instruction string."""
        prompt = StatementGeneratorPrompt()
        assert hasattr(prompt, "instruction")
        assert isinstance(prompt.instruction, str)
        assert len(prompt.instruction) > 0

    def test_statement_faithfulness_prompt_has_instruction(self):
        """StatementFaithfulnessPrompt should have an instruction string."""
        prompt = StatementFaithfulnessPrompt()
        assert hasattr(prompt, "instruction")
        assert isinstance(prompt.instruction, str)
        assert len(prompt.instruction) > 0

    def test_prompts_can_generate_string(self):
        """Prompts should be able to generate prompt strings via to_string()."""
        from ragas.metrics.collections.noise_sensitivity.util import (
            StatementFaithfulnessInput,
            StatementGeneratorInput,
        )

        gen_prompt = StatementGeneratorPrompt()
        gen_input = StatementGeneratorInput(question="What is X?", text="X is Y.")
        gen_string = gen_prompt.to_string(gen_input)
        assert isinstance(gen_string, str)
        assert len(gen_string) > 0

        faith_prompt = StatementFaithfulnessPrompt()
        faith_input = StatementFaithfulnessInput(
            context="Some context", statements=["Statement 1"]
        )
        faith_string = faith_prompt.to_string(faith_input)
        assert isinstance(faith_string, str)
        assert len(faith_string) > 0


class TestNoiseSensitivityMetricPromptMixin:
    """Tests for NoiseSensitivity PromptMixin integration."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock InstructorBaseRagasLLM for testing."""
        from unittest.mock import MagicMock

        from ragas.llms.base import InstructorBaseRagasLLM

        mock = MagicMock(spec=InstructorBaseRagasLLM)
        return mock

    def test_noise_sensitivity_has_prompt_mixin(self, mock_llm):
        """NoiseSensitivity should have PromptMixin methods."""
        from ragas.metrics.collections import NoiseSensitivity
        from ragas.prompt import PromptMixin

        metric = NoiseSensitivity(llm=mock_llm)
        assert isinstance(metric, PromptMixin)

    def test_noise_sensitivity_get_prompts(self, mock_llm):
        """NoiseSensitivity.get_prompts() should return the prompts."""
        from ragas.metrics.collections import NoiseSensitivity

        metric = NoiseSensitivity(llm=mock_llm)
        prompts = metric.get_prompts()

        assert isinstance(prompts, dict)
        assert len(prompts) == 2
        # Check that prompts are PydanticPrompt instances
        for name, prompt in prompts.items():
            assert isinstance(prompt, PydanticPrompt)

    def test_noise_sensitivity_prompts_have_names(self, mock_llm):
        """NoiseSensitivity prompts should have proper names."""
        from ragas.metrics.collections import NoiseSensitivity

        metric = NoiseSensitivity(llm=mock_llm)
        prompts = metric.get_prompts()

        # All prompts should have names
        for name, prompt in prompts.items():
            assert hasattr(prompt, "name")
            assert isinstance(prompt.name, str)

    def test_noise_sensitivity_has_adapt_prompts_method(self, mock_llm):
        """NoiseSensitivity should have adapt_prompts method."""
        from ragas.metrics.collections import NoiseSensitivity

        metric = NoiseSensitivity(llm=mock_llm)
        assert hasattr(metric, "adapt_prompts")
        assert callable(metric.adapt_prompts)

    def test_noise_sensitivity_prompts_support_adapt(self, mock_llm):
        """NoiseSensitivity prompts should support the adapt method."""
        from ragas.metrics.collections import NoiseSensitivity

        metric = NoiseSensitivity(llm=mock_llm)
        prompts = metric.get_prompts()

        for name, prompt in prompts.items():
            assert hasattr(prompt, "adapt")
            assert callable(prompt.adapt)
            assert hasattr(prompt, "examples")
