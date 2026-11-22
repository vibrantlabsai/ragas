"""Noise Sensitivity prompt classes and models."""

import typing as t

from pydantic import BaseModel, Field

from ragas.prompt import PydanticPrompt


class StatementGeneratorInput(BaseModel):
    """Input for statement generation."""

    question: str = Field(..., description="The question asked")
    text: str = Field(..., description="The text to decompose into statements")


class StatementGeneratorOutput(BaseModel):
    """Output from statement generation."""

    statements: t.List[str] = Field(..., description="Generated statements")


class StatementGeneratorPrompt(
    PydanticPrompt[StatementGeneratorInput, StatementGeneratorOutput]
):
    """Prompt for decomposing text into atomic statements."""

    instruction = "Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Format the outputs in JSON."
    input_model = StatementGeneratorInput
    output_model = StatementGeneratorOutput
    examples = [
        (
            StatementGeneratorInput(
                question="Who was Albert Einstein and what is he best known for?",
                text="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
            ),
            StatementGeneratorOutput(
                statements=[
                    "Albert Einstein was a German-born theoretical physicist.",
                    "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
                    "Albert Einstein was best known for developing the theory of relativity.",
                    "Albert Einstein also made important contributions to the development of the theory of quantum mechanics.",
                ]
            ),
        )
    ]


class StatementFaithfulnessInput(BaseModel):
    """Input for NLI statement evaluation."""

    context: str = Field(..., description="The context to verify against")
    statements: t.List[str] = Field(..., description="The statements to verify")


class StatementFaithfulnessAnswer(BaseModel):
    """Individual statement with reason and verdict for NLI evaluation."""

    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")


class StatementFaithfulnessOutput(BaseModel):
    """Output from NLI statement evaluation."""

    statements: t.List[StatementFaithfulnessAnswer]


class StatementFaithfulnessPrompt(
    PydanticPrompt[StatementFaithfulnessInput, StatementFaithfulnessOutput]
):
    """Prompt for verifying statement faithfulness using NLI."""

    instruction = "Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context."
    input_model = StatementFaithfulnessInput
    output_model = StatementFaithfulnessOutput
    examples = [
        (
            StatementFaithfulnessInput(
                context="John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.",
                statements=[
                    "John is majoring in Biology.",
                    "John is taking a course on Artificial Intelligence.",
                    "John is a dedicated student.",
                    "John has a part-time job.",
                ],
            ),
            StatementFaithfulnessOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="John is majoring in Biology.",
                        reason="John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is taking a course on Artificial Intelligence.",
                        reason="The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is a dedicated student.",
                        reason="The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John has a part-time job.",
                        reason="There is no information given in the context about John having a part-time job.",
                        verdict=0,
                    ),
                ]
            ),
        ),
        (
            StatementFaithfulnessInput(
                context="Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
                statements=[
                    "Albert Einstein was a genius.",
                ],
            ),
            StatementFaithfulnessOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Albert Einstein was a genius.",
                        reason="The context and statement are unrelated",
                        verdict=0,
                    )
                ]
            ),
        ),
    ]
