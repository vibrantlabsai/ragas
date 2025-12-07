"""Faithfulness metric v2 - Modern implementation with multi-step pipeline."""

import typing as t
from typing import List

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import (
    NLIStatementInput,
    NLIStatementOutput,
    NLIStatementPrompt,
    StatementGeneratorInput,
    StatementGeneratorOutput,
    StatementGeneratorPrompt,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class Faithfulness(BaseMetric):
    """
    Faithfulness metric using multi-step pipeline evaluation.

    Measures how factually consistent a response is with the retrieved context.
    A response is considered faithful if all its claims can be supported by the context.

    The metric works by:
    1. Breaking down the response into atomic statements
    2. Checking each statement against the retrieved contexts using NLI
    3. Computing faithfulness as the ratio of supported statements

    This implementation uses modern instructor LLMs with structured output.
    Only supports modern components - legacy wrappers are rejected with clear error messages.

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import Faithfulness
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = Faithfulness(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="Where was Einstein born?",
        ...     response="Einstein was born in Germany on 14th March 1879.",
        ...     retrieved_contexts=["Albert Einstein was born in Germany..."]
        ... )
        >>> print(f"Faithfulness Score: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for statement generation and NLI evaluation
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "faithfulness",
        **kwargs,
    ):
        """
        Initialize Faithfulness metric with required components.

        Args:
            llm: Modern instructor-based LLM for statement generation and NLI evaluation
            name: The metric name
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.statement_generator_prompt = StatementGeneratorPrompt()
        self.nli_statement_prompt = NLIStatementPrompt()

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(
        self, user_input: str, response: str, retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Calculate faithfulness score using multi-step pipeline.

        Args:
            user_input: The original question
            response: The response to evaluate for faithfulness
            retrieved_contexts: The retrieved contexts to check against

        Returns:
            MetricResult with faithfulness score (0.0-1.0, higher is better)
        """
        # Input validation
        if not response:
            raise ValueError(
                "response is missing. Please add response to the test sample."
            )
        if not user_input:
            raise ValueError(
                "user_input is missing. Please add user_input to the test sample."
            )
        if not retrieved_contexts:
            raise ValueError(
                "retrieved_contexts is missing. Please add retrieved_contexts to the test sample."
            )

        # Step 1: Break response into atomic statements
        statements = await self._create_statements(user_input, response)

        if not statements:
            # No statements generated - return NaN like legacy
            return MetricResult(value=float("nan"))

        # Step 2: Join all contexts and evaluate statements against them
        context_str = "\n".join(retrieved_contexts)
        verdicts = await self._create_verdicts(statements, context_str)

        # Step 3: Compute faithfulness score
        score = self._compute_score(verdicts)

        return MetricResult(value=float(score))

    async def _create_statements(self, question: str, response: str) -> List[str]:
        """Break response into atomic statements using statement generator."""
        input_data = StatementGeneratorInput(question=question, answer=response)
        prompt_str = self.statement_generator_prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_str, StatementGeneratorOutput)
        return result.statements

    async def _create_verdicts(
        self, statements: List[str], context: str
    ) -> NLIStatementOutput:
        """Evaluate statement faithfulness against context using NLI."""
        input_data = NLIStatementInput(context=context, statements=statements)
        prompt_str = self.nli_statement_prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_str, NLIStatementOutput)
        return result

    def _compute_score(self, verdicts: NLIStatementOutput) -> float:
        """Compute faithfulness score as ratio of faithful statements."""
        if not verdicts.statements:
            return float("nan")

        faithful_statements = sum(
            1 if statement.verdict else 0 for statement in verdicts.statements
        )
        num_statements = len(verdicts.statements)

        if num_statements > 0:
            score = faithful_statements / num_statements
        else:
            score = float("nan")

        return score
