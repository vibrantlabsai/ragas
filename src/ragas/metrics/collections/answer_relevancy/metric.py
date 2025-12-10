"""Answer Relevancy metrics v2 - Modern implementation with structured prompts."""

import typing as t

import numpy as np

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import (
    AnswerRelevanceInput,
    AnswerRelevanceOutput,
    AnswerRelevancePrompt,
)

if t.TYPE_CHECKING:
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM


class AnswerRelevancy(BaseMetric):
    """
    Modern v2 implementation of answer relevancy evaluation.

    Evaluates answer relevancy by generating multiple questions from the response
    and comparing them to the original question using cosine similarity.
    The metric detects evasive/noncommittal answers.

    This implementation uses modern instructor LLMs with structured output
    and modern embeddings for semantic comparison.
    Only supports modern components - legacy wrappers are rejected with clear error messages.

    Usage:
        >>> import openai
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.embeddings.base import embedding_factory
        >>> from ragas.metrics.collections import AnswerRelevancy
        >>>
        >>> # Setup dependencies
        >>> client = openai.AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>> embeddings = embedding_factory("openai", model="text-embedding-ada-002", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=3)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     response="Paris is the capital of France."
        ... )
        >>> print(f"Answer Relevancy: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for question generation
        embeddings: Modern embeddings model for semantic comparison
        name: The metric name
        strictness: Number of questions to generate (default: 3)
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"
    embeddings: "BaseRagasEmbedding"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        embeddings: "BaseRagasEmbedding",
        name: str = "answer_relevancy",
        strictness: int = 3,
        **kwargs,
    ):
        """
        Initialize AnswerRelevancy metric with required components.

        Args:
            llm: Modern instructor-based LLM for question generation
            embeddings: Modern embeddings model for semantic comparison
            name: The metric name (default: "answer_relevancy")
            strictness: Number of questions to generate (default: 3)
            **kwargs: Additional arguments passed to BaseMetric
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.embeddings = embeddings
        self.strictness = strictness
        self.prompt = AnswerRelevancePrompt()  # Initialize prompt class once

        # Call super() for validation
        super().__init__(name=name, **kwargs)

    async def ascore(self, user_input: str, response: str) -> MetricResult:
        """
        Calculate answer relevancy score asynchronously.

        Components are guaranteed to be validated and non-None by the base class.

        Args:
            user_input: The original question
            response: The response to evaluate

        Returns:
            MetricResult with relevancy score (0.0-1.0, higher is better)
        """
        # Input validation
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not response:
            raise ValueError("response cannot be empty")

        # Generate multiple questions from response
        generated_questions = []
        noncommittal_flags = []

        for _ in range(self.strictness):
            # Create input data and generate prompt
            input_data = AnswerRelevanceInput(response=response)
            prompt_string = self.prompt.to_string(input_data)
            result = await self.llm.agenerate(prompt_string, AnswerRelevanceOutput)

            if result.question:
                generated_questions.append(result.question)
                noncommittal_flags.append(result.noncommittal)

        if not generated_questions:
            return MetricResult(value=0.0)

        # Check if all responses are noncommittal
        all_noncommittal = np.all(noncommittal_flags)

        # Embed the original question
        question_vec = np.asarray(
            await self.embeddings.aembed_text(user_input)
        ).reshape(1, -1)

        # Embed the generated questions
        gen_question_vec = np.asarray(
            await self.embeddings.aembed_texts(generated_questions)
        ).reshape(len(generated_questions), -1)

        # Calculate cosine similarity
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        cosine_sim = (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )

        # Score is average cosine similarity, reduced to 0 if response is noncommittal
        score = cosine_sim.mean() * int(not all_noncommittal)

        return MetricResult(value=float(score))
