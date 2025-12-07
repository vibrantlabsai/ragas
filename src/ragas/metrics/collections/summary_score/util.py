"""Summary Score prompt classes and models."""

import typing as t

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt


class ExtractedKeyphrasesInput(BaseModel):
    """Input model for keyphrase extraction."""

    text: str = Field(..., description="The text to extract keyphrases from")


class ExtractedKeyphrases(BaseModel):
    """Structured output for keyphrase extraction."""

    keyphrases: t.List[str] = Field(..., description="The extracted keyphrases")


class ExtractKeyphrasesPrompt(
    BasePrompt[ExtractedKeyphrasesInput, ExtractedKeyphrases]
):
    """Prompt for extracting keyphrases from text."""

    input_model = ExtractedKeyphrasesInput
    output_model = ExtractedKeyphrases

    instruction = """Extract keyphrases of type: Person, Organization, Location, Date/Time, Monetary Values, and Percentages."""

    examples = [
        (
            ExtractedKeyphrasesInput(
                text="Apple Inc. is a technology company based in Cupertino, California. Founded by Steve Jobs in 1976, it reached a market capitalization of $3 trillion in 2023."
            ),
            ExtractedKeyphrases(
                keyphrases=[
                    "Apple Inc.",
                    "Cupertino, California",
                    "Steve Jobs",
                    "1976",
                    "$3 trillion",
                    "2023",
                ]
            ),
        ),
    ]


class GenerateQuestionsInput(BaseModel):
    """Input model for question generation."""

    text: str = Field(..., description="The text to generate questions about")
    keyphrases: t.List[str] = Field(
        ..., description="The keyphrases to base questions on"
    )


class QuestionsGenerated(BaseModel):
    """Structured output for question generation."""

    questions: t.List[str] = Field(..., description="The generated questions")


class GenerateQuestionsPrompt(BasePrompt[GenerateQuestionsInput, QuestionsGenerated]):
    """Prompt for generating questions from keyphrases."""

    input_model = GenerateQuestionsInput
    output_model = QuestionsGenerated

    instruction = """Based on the given text and keyphrases, generate closed-ended questions that can be answered with '1' if the question can be answered using the text, or '0' if it cannot. The questions should ALWAYS result in a '1' based on the given text."""

    examples = [
        (
            GenerateQuestionsInput(
                text="Apple Inc. is a technology company based in Cupertino, California. Founded by Steve Jobs in 1976, it reached a market capitalization of $3 trillion in 2023.",
                keyphrases=[
                    "Apple Inc.",
                    "Cupertino, California",
                    "Steve Jobs",
                    "1976",
                    "$3 trillion",
                    "2023",
                ],
            ),
            QuestionsGenerated(
                questions=[
                    "Is Apple Inc. a technology company?",
                    "Is Apple Inc. based in Cupertino, California?",
                    "Was Apple Inc. founded by Steve Jobs?",
                    "Was Apple Inc. founded in 1976?",
                    "Did Apple Inc. reach a market capitalization of $3 trillion?",
                    "Did Apple Inc. reach a market capitalization of $3 trillion in 2023?",
                ]
            ),
        ),
    ]


class GenerateAnswersInput(BaseModel):
    """Input model for answer generation."""

    summary: str = Field(..., description="The summary to evaluate")
    questions: t.List[str] = Field(
        ..., description="The questions to check against the summary"
    )


class AnswersGenerated(BaseModel):
    """Structured output for answer generation."""

    answers: t.List[str] = Field(
        ..., description="The answers ('0' or '1' for each question)"
    )


class GenerateAnswersPrompt(BasePrompt[GenerateAnswersInput, AnswersGenerated]):
    """Prompt for checking if summary answers questions."""

    input_model = GenerateAnswersInput
    output_model = AnswersGenerated

    instruction = """Based on the list of close-ended '1' or '0' questions, generate a JSON with key 'answers', which is a list of strings that determines whether the provided summary contains sufficient information to answer EACH question. Answers should STRICTLY be either '1' or '0'. Answer '0' if the provided summary does not contain enough information to answer the question and answer '1' if the provided summary can answer the question."""

    examples = [
        (
            GenerateAnswersInput(
                summary="Apple Inc. is a technology company based in Cupertino, California. Founded by Steve Jobs in 1976, it reached a market capitalization of $3 trillion in 2023.",
                questions=[
                    "Is Apple Inc. a technology company?",
                    "Is Apple Inc. based in Cupertino, California?",
                    "Was Apple Inc. founded by Steve Jobs?",
                    "Was Apple Inc. founded in 1976?",
                    "Did Apple Inc. reach a market capitalization of $3 trillion?",
                    "Did Apple Inc. reach a market capitalization of $3 trillion in 2023?",
                    "Is Apple Inc. a major software company?",
                    "Is Apple Inc. known for the iPhone?",
                    "Was Steve Jobs the co-founder of Apple Inc.?",
                ],
            ),
            AnswersGenerated(answers=["1", "1", "1", "1", "1", "1", "0", "0", "1"]),
        ),
    ]
