"""Answer Relevancy prompt classes and models."""

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt


class AnswerRelevanceInput(BaseModel):
    """Input model for answer relevance evaluation."""

    response: str = Field(
        ..., description="The response/answer to generate questions from"
    )


class AnswerRelevanceOutput(BaseModel):
    """Structured output for answer relevance question generation."""

    question: str = Field(
        ..., description="Question that can be answered from the response"
    )
    noncommittal: int = Field(
        ...,
        description="1 if the response is evasive/vague, 0 if it is substantive",
    )


class AnswerRelevancePrompt(BasePrompt[AnswerRelevanceInput, AnswerRelevanceOutput]):
    """Answer relevance evaluation prompt with structured input/output."""

    input_model = AnswerRelevanceInput
    output_model = AnswerRelevanceOutput

    instruction = """Generate a question for the given answer and identify if the answer is noncommittal.
Give noncommittal as 1 if the answer is noncommittal (evasive, vague, or ambiguous) and 0 if the answer is substantive.
Examples of noncommittal answers: "I don't know", "I'm not sure", "It depends"."""

    examples = [
        (
            AnswerRelevanceInput(response="Albert Einstein was born in Germany."),
            AnswerRelevanceOutput(
                question="Where was Albert Einstein born?",
                noncommittal=0,
            ),
        ),
        (
            AnswerRelevanceInput(
                response="The capital of France is Paris, a city known for its architecture and culture."
            ),
            AnswerRelevanceOutput(
                question="What is the capital of France?",
                noncommittal=0,
            ),
        ),
        (
            AnswerRelevanceInput(
                response="I don't know about the groundbreaking feature of the smartphone invented in 2023 as I am unaware of information beyond 2022."
            ),
            AnswerRelevanceOutput(
                question="What was the groundbreaking feature of the smartphone invented in 2023?",
                noncommittal=1,
            ),
        ),
    ]
