"""InstanceSpecificRubrics prompt classes and models."""

import typing as t

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt


class InstanceRubricScoreInput(BaseModel):
    """Input model for instance-specific rubric scoring."""

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
    rubrics: t.Dict[str, str] = Field(
        ..., description="The scoring rubrics for this specific instance"
    )


class InstanceRubricScoreOutput(BaseModel):
    """Output model for instance-specific rubric scoring."""

    feedback: str = Field(..., description="Detailed feedback explaining the score")
    score: int = Field(..., description="Score based on the provided rubric")


class InstanceRubricScorePrompt(
    BasePrompt[InstanceRubricScoreInput, InstanceRubricScoreOutput]
):
    """Prompt for scoring responses using instance-specific rubrics."""

    input_model = InstanceRubricScoreInput
    output_model = InstanceRubricScoreOutput

    instruction = "Your task is to assign an appropriate score and provide feedback to the inputs based solely on the scoring criteria passed in the input."

    examples = [
        (
            InstanceRubricScoreInput(
                user_input="Write a professional email to decline a meeting invitation.",
                response="Hi, I can't make it to the meeting. Sorry about that.",
                rubrics={
                    "score1_description": "The email is unprofessional, rude, or inappropriate.",
                    "score2_description": "The email lacks professionalism and misses key elements like proper greeting or closing.",
                    "score3_description": "The email is somewhat professional but could be more polished or complete.",
                    "score4_description": "The email is professional with minor room for improvement.",
                    "score5_description": "The email is highly professional, courteous, and complete.",
                },
            ),
            InstanceRubricScoreOutput(
                feedback="The response is too casual for a professional email. It lacks a proper greeting, professional tone, and courteous closing. While it conveys the message, it doesn't meet professional email standards.",
                score=2,
            ),
        ),
        (
            InstanceRubricScoreInput(
                user_input="Explain the concept of machine learning to a 10-year-old.",
                response="Machine learning is like teaching a computer to learn from examples, just like how you learn to recognize animals by seeing many pictures of them. The computer looks at lots of examples and figures out patterns on its own!",
                rubrics={
                    "score1_description": "The explanation is too complex or uses jargon inappropriate for a child.",
                    "score2_description": "The explanation has some child-friendly elements but is mostly confusing.",
                    "score3_description": "The explanation is understandable but could use better analogies.",
                    "score4_description": "The explanation is clear and mostly appropriate for a 10-year-old.",
                    "score5_description": "The explanation is perfectly tailored for a 10-year-old with great analogies.",
                },
            ),
            InstanceRubricScoreOutput(
                feedback="The response uses an excellent analogy comparing machine learning to how children learn to recognize animals. The language is age-appropriate and the concept is clearly explained without technical jargon.",
                score=5,
            ),
        ),
    ]
