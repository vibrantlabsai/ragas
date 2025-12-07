"""Context Recall prompt classes and models."""

from typing import List

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt


class ContextRecallInput(BaseModel):
    """Input model for context recall evaluation."""

    question: str = Field(..., description="The original question asked by the user")
    context: str = Field(..., description="The retrieved context passage to evaluate")
    answer: str = Field(
        ..., description="The reference answer containing statements to classify"
    )


class ContextRecallClassification(BaseModel):
    """Classification of a single statement."""

    statement: str = Field(
        ..., description="Individual statement extracted from the answer"
    )
    reason: str = Field(
        ...,
        description="Reasoning for why the statement is or isn't attributable to context",
    )
    attributed: int = Field(
        ...,
        description="Binary classification: 1 if the statement can be attributed to context, 0 otherwise",
    )


class ContextRecallOutput(BaseModel):
    """Structured output for context recall classifications."""

    classifications: List[ContextRecallClassification] = Field(
        ..., description="List of statement classifications"
    )


class ContextRecallPrompt(BasePrompt[ContextRecallInput, ContextRecallOutput]):
    """Context recall evaluation prompt with structured input/output."""

    input_model = ContextRecallInput
    output_model = ContextRecallOutput

    instruction = """Given a context and an answer, analyze each statement in the answer and classify if the statement can be attributed to the given context or not.
Use only binary classification: 1 if the statement can be attributed to the context, 0 if it cannot.
Provide detailed reasoning for each classification."""

    examples = [
        (
            ContextRecallInput(
                question="What can you tell me about Albert Einstein?",
                context="Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass-energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.",
                answer="Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895.",
            ),
            ContextRecallOutput(
                classifications=[
                    ContextRecallClassification(
                        statement="Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.",
                        reason="The date of birth of Einstein is mentioned clearly in the context.",
                        attributed=1,
                    ),
                    ContextRecallClassification(
                        statement="He received the 1921 Nobel Prize in Physics for his services to theoretical physics.",
                        reason="The exact sentence is present in the given context.",
                        attributed=1,
                    ),
                    ContextRecallClassification(
                        statement="He published 4 papers in 1905.",
                        reason="There is no mention about papers he wrote in the given context.",
                        attributed=0,
                    ),
                    ContextRecallClassification(
                        statement="Einstein moved to Switzerland in 1895.",
                        reason="There is no supporting evidence for this in the given context.",
                        attributed=0,
                    ),
                ]
            ),
        ),
        (
            ContextRecallInput(
                question="who won 2020 icc world cup?",
                context="The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.",
                answer="England",
            ),
            ContextRecallOutput(
                classifications=[
                    ContextRecallClassification(
                        statement="England",
                        reason="The context clarifies that England won the 2022 edition (which was originally scheduled for 2020).",
                        attributed=1,
                    ),
                ]
            ),
        ),
        (
            ContextRecallInput(
                question="What is the tallest mountain in the world?",
                context="The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.",
                answer="Mount Everest.",
            ),
            ContextRecallOutput(
                classifications=[
                    ContextRecallClassification(
                        statement="Mount Everest.",
                        reason="The provided context discusses the Andes mountain range, which does not include Mount Everest or directly relate to the world's tallest mountain.",
                        attributed=0,
                    ),
                ]
            ),
        ),
    ]
