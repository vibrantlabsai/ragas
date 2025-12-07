"""Answer Accuracy prompt classes and models."""

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt


class AnswerAccuracyInput(BaseModel):
    """Input model for answer accuracy evaluation."""

    query: str = Field(..., description="The original question")
    user_answer: str = Field(..., description="The user's answer to evaluate")
    reference_answer: str = Field(..., description="The ground truth reference answer")


class AnswerAccuracyOutput(BaseModel):
    """Structured output for answer accuracy evaluation."""

    rating: int = Field(..., description="Accuracy rating (0, 2, or 4)")


class AnswerAccuracyJudge1Prompt(BasePrompt[AnswerAccuracyInput, AnswerAccuracyOutput]):
    """First judge prompt for answer accuracy evaluation."""

    input_model = AnswerAccuracyInput
    output_model = AnswerAccuracyOutput

    instruction = """You are a world class state of the art assistant for rating a User Answer given a Question. The Question is completely answered by the Reference Answer.
Say 4, if User Answer is full contained and equivalent to Reference Answer in all terms, topics, numbers, metrics, dates and units.
Say 2, if User Answer is partially contained and almost equivalent to Reference Answer in all terms, topics, numbers, metrics, dates and units.
Say 0, if User Answer is not contained in Reference Answer or not accurate in all terms, topics, numbers, metrics, dates and units or the User Answer do not answer the question.
Do not explain or justify your rating. Your rating must be only 4, 2 or 0 according to the instructions above.
Return your response as JSON in this format: {"rating": X} where X is 0, 2, or 4."""

    examples = [
        (
            AnswerAccuracyInput(
                query="When was Albert Einstein born?",
                user_answer="Albert Einstein was born in 1879.",
                reference_answer="Albert Einstein was born on March 14, 1879.",
            ),
            AnswerAccuracyOutput(rating=2),
        ),
        (
            AnswerAccuracyInput(
                query="What is the capital of France?",
                user_answer="Paris is the capital of France.",
                reference_answer="Paris is the capital of France.",
            ),
            AnswerAccuracyOutput(rating=4),
        ),
        (
            AnswerAccuracyInput(
                query="What is the highest mountain?",
                user_answer="The Eiffel Tower is a famous landmark.",
                reference_answer="Mount Everest is the highest mountain.",
            ),
            AnswerAccuracyOutput(rating=0),
        ),
    ]


class AnswerAccuracyJudge2Prompt(BasePrompt[AnswerAccuracyInput, AnswerAccuracyOutput]):
    """Second judge prompt for answer accuracy evaluation."""

    input_model = AnswerAccuracyInput
    output_model = AnswerAccuracyOutput

    instruction = """I will rate the User Answer in comparison to the Reference Answer for a given Question.
A rating of 4 indicates that the User Answer is entirely consistent with the Reference Answer, covering all aspects, topics, numbers, metrics, dates, and units.
A rating of 2 signifies that the User Answer is mostly aligned with the Reference Answer, with minor discrepancies in some areas.
A rating of 0 means that the User Answer is either inaccurate, incomplete, or unrelated to the Reference Answer, or it fails to address the Question.
I will provide the rating without any explanation or justification, adhering to the following scale: 0 (no match), 2 (partial match), 4 (exact match).
Do not explain or justify my rating. My rating must be only 4, 2 or 0 only.
Return your response as JSON in this format: {"rating": X} where X is 0, 2, or 4."""

    examples = [
        (
            AnswerAccuracyInput(
                query="When was Albert Einstein born?",
                user_answer="Einstein was born in 1879 in Germany.",
                reference_answer="Albert Einstein was born on March 14, 1879 in Ulm, Germany.",
            ),
            AnswerAccuracyOutput(rating=2),
        ),
        (
            AnswerAccuracyInput(
                query="What is the capital of France?",
                user_answer="The capital of France is Paris.",
                reference_answer="Paris is the capital of France.",
            ),
            AnswerAccuracyOutput(rating=4),
        ),
        (
            AnswerAccuracyInput(
                query="What is the speed of light?",
                user_answer="The sun is a star.",
                reference_answer="The speed of light is approximately 299,792,458 meters per second.",
            ),
            AnswerAccuracyOutput(rating=0),
        ),
    ]
