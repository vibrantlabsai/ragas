"""Summary Score metric v2 - Modern implementation with multi-step pipeline."""

import logging
import typing as t
from typing import List

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import (
    AnswersGenerated,
    ExtractedKeyphrases,
    ExtractedKeyphrasesInput,
    ExtractKeyphrasesPrompt,
    GenerateAnswersInput,
    GenerateAnswersPrompt,
    GenerateQuestionsInput,
    GenerateQuestionsPrompt,
    QuestionsGenerated,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class SummaryScore(BaseMetric):
    """
    Summary Score metric using multi-step pipeline evaluation.

    Measures how well a summary captures important information from contexts by:
    1. Extracting keyphrases from the original contexts
    2. Generating yes/no questions from those keyphrases
    3. Checking if the summary can answer those questions
    4. Optionally penalizing overly long summaries for conciseness

    This implementation uses modern instructor LLMs with structured output.
    Only supports modern components - legacy wrappers are rejected with clear error messages.

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import SummaryScore
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = SummaryScore(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     reference_contexts=["Apple Inc. is a technology company..."],
        ...     response="Apple is a tech company founded by Steve Jobs."
        ... )
        >>> print(f"Summary Score: {result.value}")
        >>>
        >>> # Custom configuration (more conciseness focus)
        >>> concise_metric = SummaryScore(
        ...     llm=llm,
        ...     length_penalty=True,
        ...     coeff=0.8  # More weight on conciseness
        ... )

    Attributes:
        llm: Modern instructor-based LLM for keyphrase, question, and answer generation
        name: The metric name
        length_penalty: Whether to apply conciseness penalty for long summaries
        coeff: Weight for conciseness score (0.0=only QA, 1.0=only conciseness)
        allowed_values: Score range (0.0 to 1.0)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "summary_score",
        length_penalty: bool = True,
        coeff: float = 0.5,
        **kwargs,
    ):
        """
        Initialize SummaryScore metric with required components.

        Args:
            llm: Modern instructor-based LLM for keyphrase, question, and answer generation
            name: The metric name
            length_penalty: Whether to apply conciseness penalty for long summaries
            coeff: Weight for conciseness score (0.0=only QA, 1.0=only conciseness)
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.length_penalty = length_penalty
        self.coeff = coeff
        self.extract_keyphrases_prompt = ExtractKeyphrasesPrompt()
        self.generate_questions_prompt = GenerateQuestionsPrompt()
        self.generate_answers_prompt = GenerateAnswersPrompt()

        # Validate coefficient
        if not (0.0 <= coeff <= 1.0):
            raise ValueError(f"Coefficient must be between 0.0 and 1.0, got {coeff}")

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(
        self, reference_contexts: List[str], response: str
    ) -> MetricResult:
        """
        Calculate summary score using multi-step pipeline.

        Args:
            reference_contexts: The original contexts that were summarized
            response: The summary to evaluate

        Returns:
            MetricResult with summary score (0.0-1.0)

        Raises:
            ValueError: If reference_contexts is empty or response is empty/whitespace only
        """
        # Input validation
        if not reference_contexts or not any(ctx.strip() for ctx in reference_contexts):
            raise ValueError(
                "reference_contexts cannot be empty or contain only whitespace"
            )

        if not response or not response.strip():
            raise ValueError("response cannot be empty or whitespace only")

        # Step 1: Combine contexts and extract keyphrases
        text = "\n".join(reference_contexts)
        keyphrases = await self._extract_keyphrases(text)

        if not keyphrases:
            # Match legacy behavior: log error and continue with empty list
            logging.error("No keyphrases generated, unable to calculate the score.")
            keyphrases = []

        # Step 2: Generate questions from keyphrases
        questions = await self._generate_questions(text, keyphrases)

        if not questions:
            # Match legacy behavior: log error and continue with empty list
            logging.error("No questions generated, unable to calculate the score.")
            questions = []

        # Step 3: Check if summary can answer the questions
        answers = await self._generate_answers(response, questions)

        # Step 4: Calculate QA score
        qa_score = self._compute_qa_score(answers)

        # Step 5: Calculate final score (with optional conciseness penalty)
        if self.length_penalty:
            conciseness_score = self._compute_conciseness_score(text, response)
            final_score = qa_score * (1 - self.coeff) + conciseness_score * self.coeff
        else:
            final_score = qa_score

        return MetricResult(value=float(final_score))

    async def _extract_keyphrases(self, text: str) -> List[str]:
        """Extract keyphrases from text using the keyphrase extraction prompt."""
        input_data = ExtractedKeyphrasesInput(text=text)
        prompt_str = self.extract_keyphrases_prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_str, ExtractedKeyphrases)
        return result.keyphrases

    async def _generate_questions(self, text: str, keyphrases: List[str]) -> List[str]:
        """Generate questions from text and keyphrases."""
        input_data = GenerateQuestionsInput(text=text, keyphrases=keyphrases)
        prompt_str = self.generate_questions_prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_str, QuestionsGenerated)
        return result.questions

    async def _generate_answers(self, summary: str, questions: List[str]) -> List[str]:
        """Generate answers by checking if summary can answer questions."""
        input_data = GenerateAnswersInput(summary=summary, questions=questions)
        prompt_str = self.generate_answers_prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_str, AnswersGenerated)
        return result.answers

    def _compute_qa_score(self, answers: List[str]) -> float:
        """Compute QA score as ratio of correct answers. Matches legacy behavior exactly."""
        correct = sum([1 for a in answers if a.lower() == "1"])
        return correct / len(
            answers
        )  # Will raise ZeroDivisionError if answers is empty (legacy behavior)

    def _compute_conciseness_score(self, text: str, summary: str) -> float:
        """Compute conciseness score based on length ratio."""
        return 1 - min(len(summary), len(text)) / (len(text) + 1e-10)
