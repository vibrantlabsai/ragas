"""Context Relevance prompt classes and models."""

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt


class ContextRelevanceInput(BaseModel):
    """Input model for context relevance evaluation."""

    user_input: str = Field(..., description="The user's question")
    context: str = Field(..., description="The context to evaluate for relevance")


class ContextRelevanceOutput(BaseModel):
    """Structured output for context relevance evaluation."""

    rating: int = Field(..., description="Relevance rating (0, 1, or 2)")


class ContextRelevanceJudge1Prompt(
    BasePrompt[ContextRelevanceInput, ContextRelevanceOutput]
):
    """First judge prompt for context relevance evaluation."""

    input_model = ContextRelevanceInput
    output_model = ContextRelevanceOutput

    instruction = """You are a world class expert designed to evaluate the relevance score of a Context in order to answer the Question.
Your task is to determine if the Context contains proper information to answer the Question.
Do not rely on your previous knowledge about the Question.
Use only what is written in the Context and in the Question.
Follow the instructions below:
0. If the context does not contains any relevant information to answer the question, say 0.
1. If the context partially contains relevant information to answer the question, say 1.
2. If the context contains any relevant information to answer the question, say 2.
You must provide the relevance score of 0, 1, or 2, nothing else.
Do not explain.
Return your response as JSON in this format: {"rating": X} where X is 0, 1, or 2."""

    examples = [
        (
            ContextRelevanceInput(
                user_input="When was Albert Einstein born?",
                context="Albert Einstein was born March 14, 1879.",
            ),
            ContextRelevanceOutput(rating=2),
        ),
        (
            ContextRelevanceInput(
                user_input="What is photosynthesis?",
                context="Photosynthesis is the process by which plants convert sunlight into energy.",
            ),
            ContextRelevanceOutput(rating=2),
        ),
        (
            ContextRelevanceInput(
                user_input="How do computers work?",
                context="Albert Einstein was a theoretical physicist.",
            ),
            ContextRelevanceOutput(rating=0),
        ),
    ]


class ContextRelevanceJudge2Prompt(
    BasePrompt[ContextRelevanceInput, ContextRelevanceOutput]
):
    """Second judge prompt for context relevance evaluation."""

    input_model = ContextRelevanceInput
    output_model = ContextRelevanceOutput

    instruction = """As a specially designed expert to assess the relevance score of a given Context in relation to a Question, my task is to determine the extent to which the Context provides information necessary to answer the Question. I will rely solely on the information provided in the Context and Question, and not on any prior knowledge.

Here are the instructions I will follow:
* If the Context does not contain any relevant information to answer the Question, I will respond with a relevance score of 0.
* If the Context partially contains relevant information to answer the Question, I will respond with a relevance score of 1.
* If the Context contains any relevant information to answer the Question, I will respond with a relevance score of 2.
Return your response as JSON in this format: {"rating": X} where X is 0, 1, or 2."""

    examples = [
        (
            ContextRelevanceInput(
                user_input="When was Albert Einstein born?",
                context="Albert Einstein was born March 14, 1879.",
            ),
            ContextRelevanceOutput(rating=2),
        ),
        (
            ContextRelevanceInput(
                user_input="What is photosynthesis?",
                context="Photosynthesis is the process by which plants convert sunlight into energy.",
            ),
            ContextRelevanceOutput(rating=2),
        ),
        (
            ContextRelevanceInput(
                user_input="How do computers work?",
                context="The weather today is sunny.",
            ),
            ContextRelevanceOutput(rating=0),
        ),
    ]
