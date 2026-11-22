"""Faithfulness metric v2 - Modern implementation with multi-step pipeline."""

import typing as t
from typing import List

from ragas.decision_models import BaseDecisionModel, BinaryDecisionResult
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import (
    NLIStatementInput,
    NLIStatementOutput,
    NLIStatementPrompt,
    StatementFaithfulnessAnswer,
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

    Uses a modern structured-output LLM to generate statements and, by default,
    judge their support. An explicit decision_model instead judges each statement
    against the retrieved contexts. Statement generation still requires an LLM.

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
        decision_model: Optional binary judge for statement support
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "faithfulness",
        *,
        decision_model: t.Optional[BaseDecisionModel] = None,
        **kwargs,
    ):
        """
        Initialize Faithfulness metric with required components.

        Args:
            llm: Modern instructor-based LLM for statement generation and NLI evaluation
            name: The metric name
            decision_model: Optional judge receiving context and statement fields.
                Its selected value is used without rethresholding. Statements,
                verdicts, and probabilities are retained in
                result.traces["output"]["decisions"]. The NLI prompt is only used
                by the default LLM path.
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        if decision_model is not None and not isinstance(
            decision_model, BaseDecisionModel
        ):
            raise TypeError("decision_model must be a BaseDecisionModel")
        self.decision_model = decision_model
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
        decisions = []
        if self.decision_model is not None:
            answers = []
            for statement in statements:
                decision = await self.decision_model.binary(
                    instruction=(
                        "Decide whether the statement can be directly inferred "
                        "from the context."
                    ),
                    data={"context": context_str, "statement": statement},
                )
                if not isinstance(decision, BinaryDecisionResult):
                    raise TypeError(
                        "decision_model.binary() must return BinaryDecisionResult"
                    )
                decision = BinaryDecisionResult.model_validate(decision)
                decisions.append({"statement": statement, **decision.model_dump()})
                answers.append(
                    StatementFaithfulnessAnswer(
                        statement=statement,
                        reason="",  # A binary decision need not generate a rationale.
                        verdict=int(decision.value),
                    )
                )
            verdicts = NLIStatementOutput(statements=answers)
        else:
            verdicts = await self._create_verdicts(statements, context_str)

        # Step 3: Compute faithfulness score
        score = self._compute_score(verdicts)

        if self.decision_model is not None:
            return MetricResult(
                value=float(score), traces={"output": {"decisions": decisions}}
            )
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
