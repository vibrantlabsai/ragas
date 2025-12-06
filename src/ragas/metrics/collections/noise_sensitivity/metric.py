"""Noise Sensitivity metrics v2 - Modern implementation with function-based prompts."""

import typing as t
from typing import Dict, List, Literal

import numpy as np

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import (
    StatementFaithfulnessInput,
    StatementFaithfulnessOutput,
    StatementFaithfulnessPrompt,
    StatementGeneratorInput,
    StatementGeneratorOutput,
    StatementGeneratorPrompt,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class NoiseSensitivity(BaseMetric):
    """
    Modern v2 implementation of noise sensitivity evaluation.

    Measures how often a system makes errors by providing incorrect responses
    when utilizing either relevant or irrelevant retrieved documents.

    The metric works by:
    1. Decomposing reference and response into atomic statements
    2. Using NLI to evaluate statement faithfulness against each retrieved context
    3. Computing noise sensitivity based on incorrect claims from relevant/irrelevant contexts

    This implementation uses modern instructor LLMs with structured output.
    Only supports modern components - legacy wrappers are rejected with clear error messages.

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import instructor_llm_factory
        >>> from ragas.metrics.collections import NoiseSensitivity
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = instructor_llm_factory("openai", client=client, model="gpt-4o-mini")
        >>>
        >>> # Create metric instance
        >>> metric = NoiseSensitivity(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is LIC known for?",
        ...     response="LIC is the largest insurance company in India...",
        ...     reference="LIC is known for managing investments...",
        ...     retrieved_contexts=["LIC was established in 1956...", ...]
        ... )
        >>> print(f"Noise Sensitivity: {result.value}")
        >>>
        >>> # Test irrelevant context sensitivity
        >>> irrelevant_metric = NoiseSensitivity(llm=llm, mode="irrelevant")

    Attributes:
        llm: Modern instructor-based LLM for statement generation and NLI evaluation
        name: The metric name
        mode: Either "relevant" or "irrelevant" context sensitivity
        allowed_values: Score range (0.0 to 1.0, lower is better)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "noise_sensitivity",
        mode: Literal["relevant", "irrelevant"] = "relevant",
        **kwargs,
    ):
        """
        Initialize NoiseSensitivity metric with required components.

        Args:
            llm: Modern instructor-based LLM for statement generation and NLI evaluation
            name: The metric name
            mode: Either "relevant" or "irrelevant" context sensitivity mode
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.mode = mode
        self.statement_prompt = StatementGeneratorPrompt()
        self.faithfulness_prompt = StatementFaithfulnessPrompt()

        # Validate mode
        if mode not in {"relevant", "irrelevant"}:
            raise ValueError(
                f"Invalid argument passed for 'mode': {mode}. Must be 'relevant' or 'irrelevant'."
            )

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(
        self,
        user_input: str,
        response: str,
        reference: str,
        retrieved_contexts: List[str],
    ) -> MetricResult:
        """
        Calculate noise sensitivity score.

        Args:
            user_input: The original question
            response: The answer to evaluate
            reference: The ground truth reference
            retrieved_contexts: The retrieved contexts used to generate the response

        Returns:
            MetricResult with noise sensitivity score (0.0-1.0, lower is better)
        """
        # Input validation
        if not reference:
            raise ValueError(
                "reference is missing. Please add reference to the test sample."
            )
        if not user_input:
            raise ValueError(
                "user_input is missing. Please add user_input to the test sample."
            )
        if not response:
            raise ValueError(
                "response is missing. Please add response to the test sample."
            )
        if not retrieved_contexts:
            raise ValueError(
                "retrieved_contexts is missing. Please add retrieved_contexts to the test sample."
            )

        # Step 1: Decompose reference and response into statements
        gt_statements = await self._decompose_answer_into_statements(
            reference, user_input
        )
        ans_statements = await self._decompose_answer_into_statements(
            response, user_input
        )

        # Step 2: Evaluate statement faithfulness against each retrieved context
        gt_verdictslist = []
        ans_verdictslist = []

        for ctx in retrieved_contexts:
            # Evaluate ground truth statements against this context
            gt_verdicts = await self._evaluate_statement_faithfulness(
                gt_statements, ctx
            )
            gt_verdictslist.append(np.array(gt_verdicts))

            # Evaluate answer statements against this context
            ans_verdicts = await self._evaluate_statement_faithfulness(
                ans_statements, ctx
            )
            ans_verdictslist.append(np.array(ans_verdicts))

        # Step 3: Build matrices for computation (exact legacy shape handling)
        answers = {}
        answers["retrieved2ground_truth"] = np.array(gt_verdictslist).T
        answers["retrieved2answer"] = np.array(ans_verdictslist).T

        # Evaluate answer statements against reference (ground truth)
        gt_to_ans_verdicts = await self._evaluate_statement_faithfulness(
            ans_statements, reference
        )
        answers["ground_truth2answer"] = np.array(gt_to_ans_verdicts)
        # Wrap in another array to match legacy shape handling
        answers["ground_truth2answer"] = np.array([answers["ground_truth2answer"]])

        # Convert all to boolean arrays
        answers = {k: v.astype(bool) for k, v in answers.items()}

        # Step 4: Compute noise sensitivity score
        score = self._compute_score(answers)

        return MetricResult(value=float(score))

    async def _decompose_answer_into_statements(
        self, text: str, question: str
    ) -> List[str]:
        """Decompose answer text into atomic statements."""
        input_data = StatementGeneratorInput(question=question, text=text)
        prompt_str = self.statement_prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_str, StatementGeneratorOutput)
        return result.statements

    async def _evaluate_statement_faithfulness(
        self, statements: List[str], context: str
    ) -> List[int]:
        """Evaluate faithfulness of statements against context using NLI."""
        input_data = StatementFaithfulnessInput(context=context, statements=statements)
        prompt_str = self.faithfulness_prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_str, StatementFaithfulnessOutput)

        verdict_list = [
            1 if statement.verdict else 0 for statement in result.statements
        ]
        return verdict_list

    def _compute_score(self, answers: Dict) -> float:
        """Compute noise sensitivity score from faithfulness matrices."""
        incorrect = ~answers["ground_truth2answer"]

        # Compute relevant retrievals (needed for both modes)
        relevant_retrieved = np.max(
            answers["retrieved2ground_truth"], axis=0, keepdims=True
        )
        relevant_faithful = np.max(
            relevant_retrieved & answers["retrieved2answer"], axis=1
        )

        if self.mode == "irrelevant":
            # Compute irrelevant retrievals
            irrelevant_retrieved = ~relevant_retrieved
            irrelevant_faithful = np.max(
                irrelevant_retrieved & answers["retrieved2answer"], axis=1
            )

            # Keep them exclusive (irrelevant should not include relevant)
            irrelevant_faithful &= ~relevant_faithful

            return float(np.mean(irrelevant_faithful & incorrect))

        else:  # mode == "relevant"
            return float(np.mean(relevant_faithful & incorrect))
