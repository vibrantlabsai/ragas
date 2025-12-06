"""Answer Correctness prompt classes and models."""

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


class StatementsWithReason(BaseModel):
    """Individual statement with reasoning for classification."""

    statement: str = Field(..., description="The statement being classified")
    reason: str = Field(..., description="Reason for the classification")


class ClassificationWithReason(BaseModel):
    """Structured output for TP/FP/FN classification."""

    TP: t.List[StatementsWithReason] = Field(
        ..., description="True positive statements"
    )
    FP: t.List[StatementsWithReason] = Field(
        ..., description="False positive statements"
    )
    FN: t.List[StatementsWithReason] = Field(
        ..., description="False negative statements"
    )


class CorrectnessClassifierInput(BaseModel):
    """Input model for correctness classification."""

    question: str = Field(..., description="The original question")
    answer: t.List[str] = Field(..., description="Statements from the answer")
    ground_truth: t.List[str] = Field(..., description="Statements from ground truth")


class CorrectnessClassifierPrompt(
    BasePrompt[CorrectnessClassifierInput, ClassificationWithReason]
):
    """Prompt for classifying statements as TP/FP/FN."""

    input_model = CorrectnessClassifierInput
    output_model = ClassificationWithReason

    instruction = """Given a ground truth and an answer statements, analyze each statement and classify them in one of the following categories: TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth, FP (false positive): statements present in the answer but not directly supported by any statement in ground truth, FN (false negative): statements found in the ground truth but not present in answer. Each statement can only belong to one of the categories. Provide a reason for each classification."""

    examples = [
        (
            CorrectnessClassifierInput(
                question="What powers the sun and what is its primary function?",
                answer=[
                    "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                    "The primary function of the sun is to provide light to the solar system.",
                ],
                ground_truth=[
                    "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                    "This fusion process in the sun's core releases a tremendous amount of energy.",
                    "The energy from the sun provides heat and light, which are essential for life on Earth.",
                    "The sun's light plays a critical role in Earth's climate system.",
                    "Sunlight helps to drive the weather and ocean currents.",
                ],
            ),
            ClassificationWithReason(
                TP=[
                    StatementsWithReason(
                        statement="The primary function of the sun is to provide light to the solar system.",
                        reason="This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun's energy.",
                    )
                ],
                FP=[
                    StatementsWithReason(
                        statement="The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                        reason="This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion.",
                    )
                ],
                FN=[
                    StatementsWithReason(
                        statement="The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                        reason="This accurate description of the sun's power source is not included in the answer.",
                    ),
                    StatementsWithReason(
                        statement="This fusion process in the sun's core releases a tremendous amount of energy.",
                        reason="This process and its significance are not mentioned in the answer.",
                    ),
                    StatementsWithReason(
                        statement="The energy from the sun provides heat and light, which are essential for life on Earth.",
                        reason="The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers.",
                    ),
                    StatementsWithReason(
                        statement="The sun's light plays a critical role in Earth's climate system.",
                        reason="This broader impact of the sun's light on Earth's climate system is not addressed in the answer.",
                    ),
                    StatementsWithReason(
                        statement="Sunlight helps to drive the weather and ocean currents.",
                        reason="The effect of sunlight on weather patterns and ocean currents is omitted in the answer.",
                    ),
                ],
            ),
        ),
        (
            CorrectnessClassifierInput(
                question="What is the boiling point of water?",
                answer=[
                    "The boiling point of water is 100 degrees Celsius at sea level"
                ],
                ground_truth=[
                    "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
                    "The boiling point of water can change with altitude.",
                ],
            ),
            ClassificationWithReason(
                TP=[
                    StatementsWithReason(
                        statement="The boiling point of water is 100 degrees Celsius at sea level",
                        reason="This statement is directly supported by the ground truth which specifies the boiling point of water as 100 degrees Celsius at sea level.",
                    )
                ],
                FP=[],
                FN=[
                    StatementsWithReason(
                        statement="The boiling point of water can change with altitude.",
                        reason="This additional information about how the boiling point of water can vary with altitude is not mentioned in the answer.",
                    )
                ],
            ),
        ),
    ]
