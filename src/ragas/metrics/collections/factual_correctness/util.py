"""Factual Correctness prompt classes and models."""

from typing import List

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt
from ragas.prompt.metrics.factual_correctness import claim_decomposition_prompt


class ClaimDecompositionInput(BaseModel):
    """Input for claim decomposition."""

    response: str = Field(..., description="The response text to decompose into claims")
    atomicity: str = Field(
        default="low", description="Atomicity level: 'low' or 'high'"
    )
    coverage: str = Field(default="low", description="Coverage level: 'low' or 'high'")


class ClaimDecompositionOutput(BaseModel):
    """Output from claim decomposition."""

    claims: List[str] = Field(..., description="Decomposed claims")


class ClaimDecompositionPrompt(
    BasePrompt[ClaimDecompositionInput, ClaimDecompositionOutput]
):
    """Prompt for decomposing text into claims with configurable atomicity and coverage."""

    input_model = ClaimDecompositionInput
    output_model = ClaimDecompositionOutput

    def to_string(self, input_data: ClaimDecompositionInput) -> str:
        """Generate prompt string with configurable examples."""
        return claim_decomposition_prompt(
            input_data.response,
            atomicity=input_data.atomicity,
            coverage=input_data.coverage,
        )
