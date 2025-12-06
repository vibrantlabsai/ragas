"""Faithfulness prompt classes and models."""

import typing as t

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt


class StatementGeneratorInput(BaseModel):
    """Input model for statement generation."""

    question: str = Field(..., description="The question being answered")
    answer: str = Field(
        ..., description="The answer text to break down into statements"
    )


class StatementGeneratorOutput(BaseModel):
    """Structured output for statement generation."""

    statements: t.List[str] = Field(
        ..., description="The generated statements from the answer"
    )


class StatementGeneratorPrompt(
    BasePrompt[StatementGeneratorInput, StatementGeneratorOutput]
):
    """Prompt for breaking down answers into atomic statements."""

    input_model = StatementGeneratorInput
    output_model = StatementGeneratorOutput

    instruction = """Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement."""

    examples = [
        (
            StatementGeneratorInput(
                question="Who was Albert Einstein and what is he best known for?",
                answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
            ),
            StatementGeneratorOutput(
                statements=[
                    "Albert Einstein was a German-born theoretical physicist.",
                    "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
                    "Albert Einstein was best known for developing the theory of relativity.",
                    "Albert Einstein made important contributions to the development of the theory of quantum mechanics.",
                ]
            ),
        ),
    ]


class StatementFaithfulnessAnswer(BaseModel):
    """Individual statement with reason and verdict for NLI evaluation."""

    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness")


class NLIStatementInput(BaseModel):
    """Input model for NLI statement evaluation."""

    context: str = Field(..., description="The context to evaluate statements against")
    statements: t.List[str] = Field(
        ..., description="The statements to judge for faithfulness"
    )


class NLIStatementOutput(BaseModel):
    """Structured output for NLI statement evaluation."""

    statements: t.List[StatementFaithfulnessAnswer] = Field(
        ..., description="Evaluated statements with verdicts"
    )


class NLIStatementPrompt(BasePrompt[NLIStatementInput, NLIStatementOutput]):
    """Prompt for evaluating statement faithfulness against context using NLI."""

    input_model = NLIStatementInput
    output_model = NLIStatementOutput

    instruction = """Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context."""

    examples = [
        (
            NLIStatementInput(
                context="John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.",
                statements=[
                    "John is majoring in Biology.",
                    "John is taking a course on Artificial Intelligence.",
                    "John is a dedicated student.",
                    "John has a part-time job.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="John is majoring in Biology.",
                        reason="John's major is explicitly stated as Computer Science, not Biology.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is taking a course on Artificial Intelligence.",
                        reason="The context mentions courses in Data Structures, Algorithms, and Database Management, but does not mention Artificial Intelligence.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is a dedicated student.",
                        reason="The context states that John is a diligent student who spends a significant amount of time studying and completing assignments.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John has a part-time job.",
                        reason="There is no information in the context about John having a part-time job.",
                        verdict=0,
                    ),
                ]
            ),
        ),
    ]
