"""Noise Sensitivity prompt classes and models."""

from typing import List

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt
from ragas.prompt.metrics.common import nli_statement_prompt, statement_generator_prompt


class StatementGeneratorInput(BaseModel):
    """Input for statement generation."""

    question: str = Field(..., description="The question asked")
    text: str = Field(..., description="The text to decompose into statements")


class StatementGeneratorOutput(BaseModel):
    """Output from statement generation."""

    statements: List[str] = Field(..., description="Generated statements")


class StatementGeneratorPrompt(
    BasePrompt[StatementGeneratorInput, StatementGeneratorOutput]
):
    """Prompt for decomposing text into atomic statements."""

    input_model = StatementGeneratorInput
    output_model = StatementGeneratorOutput

    def to_string(self, input_data: StatementGeneratorInput) -> str:
        """Generate prompt string."""
        return statement_generator_prompt(input_data.question, input_data.text)


class StatementFaithfulnessInput(BaseModel):
    """Input for NLI statement evaluation."""

    context: str = Field(..., description="The context to verify against")
    statements: List[str] = Field(..., description="The statements to verify")


class StatementFaithfulnessAnswer(BaseModel):
    """Individual statement with reason and verdict for NLI evaluation."""

    statement: str
    reason: str
    verdict: int


class StatementFaithfulnessOutput(BaseModel):
    """Output from NLI statement evaluation."""

    statements: List[StatementFaithfulnessAnswer]


class StatementFaithfulnessPrompt(
    BasePrompt[StatementFaithfulnessInput, StatementFaithfulnessOutput]
):
    """Prompt for verifying statement faithfulness using NLI."""

    input_model = StatementFaithfulnessInput
    output_model = StatementFaithfulnessOutput

    def to_string(self, input_data: StatementFaithfulnessInput) -> str:
        """Generate prompt string."""
        return nli_statement_prompt(input_data.context, input_data.statements)
