"""DomainSpecificRubrics prompt classes and models."""

import typing as t

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt

DEFAULT_REFERENCE_FREE_RUBRICS = {
    "score1_description": "The response is entirely incorrect and fails to address any aspect of the user input.",
    "score2_description": "The response contains partial accuracy but includes major errors or significant omissions that affect its relevance to the user input.",
    "score3_description": "The response is mostly accurate but lacks clarity, thoroughness, or minor details needed to fully address the user input.",
    "score4_description": "The response is accurate and clear, with only minor omissions or slight inaccuracies in addressing the user input.",
    "score5_description": "The response is completely accurate, clear, and thoroughly addresses the user input without any errors or omissions.",
}

DEFAULT_WITH_REFERENCE_RUBRICS = {
    "score1_description": "The response is entirely incorrect, irrelevant, or does not align with the reference in any meaningful way.",
    "score2_description": "The response partially matches the reference but contains major errors, significant omissions, or irrelevant information.",
    "score3_description": "The response aligns with the reference overall but lacks sufficient detail, clarity, or contains minor inaccuracies.",
    "score4_description": "The response is mostly accurate, aligns closely with the reference, and contains only minor issues or omissions.",
    "score5_description": "The response is fully accurate, completely aligns with the reference, and is clear, thorough, and detailed.",
}


class RubricScoreInput(BaseModel):
    """Input model for rubric-based scoring."""

    user_input: t.Optional[str] = Field(
        default=None, description="The input/question provided to the system"
    )
    response: t.Optional[str] = Field(
        default=None, description="The response from the system"
    )
    retrieved_contexts: t.Optional[t.List[str]] = Field(
        default=None, description="The contexts retrieved for generating the response"
    )
    reference_contexts: t.Optional[t.List[str]] = Field(
        default=None, description="The reference contexts for evaluation"
    )
    reference: t.Optional[str] = Field(
        default=None, description="The reference/ground truth answer"
    )


class RubricScoreOutput(BaseModel):
    """Output model for rubric-based scoring."""

    feedback: str = Field(..., description="Detailed feedback explaining the score")
    score: int = Field(..., description="Score from 1-5 based on the rubric")


class RubricScorePrompt(BasePrompt[RubricScoreInput, RubricScoreOutput]):
    """Prompt for scoring responses using a rubric."""

    input_model = RubricScoreInput
    output_model = RubricScoreOutput

    instruction = "Your task is to assign an appropriate score and provide feedback to the inputs based solely on the scoring criteria."

    examples = [
        (
            RubricScoreInput(
                user_input="What is the capital of France?",
                response="The capital of France is Paris.",
                reference="Paris is the capital and largest city of France.",
            ),
            RubricScoreOutput(
                feedback="The response correctly identifies Paris as the capital of France, which fully aligns with the reference. The answer is accurate, clear, and directly addresses the question.",
                score=5,
            ),
        ),
        (
            RubricScoreInput(
                user_input="Explain photosynthesis.",
                response="Photosynthesis is when plants make food.",
                reference="Photosynthesis is the process by which plants convert light energy into chemical energy, using carbon dioxide and water to produce glucose and oxygen.",
            ),
            RubricScoreOutput(
                feedback="The response captures the basic concept that plants make food but lacks the scientific detail about light energy conversion, the role of carbon dioxide and water, and the production of glucose and oxygen. It aligns with the reference at a very high level but misses substantial detail.",
                score=3,
            ),
        ),
    ]


def format_rubrics(rubrics: t.Dict[str, str]) -> str:
    """Format rubrics dictionary into a string for the prompt."""
    return "\n".join(f"{key}: {value}" for key, value in rubrics.items())
