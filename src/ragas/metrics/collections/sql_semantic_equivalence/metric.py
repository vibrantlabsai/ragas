"""SQLSemanticEquivalence metric - Modern collections implementation."""

import typing as t
from typing import List, Optional

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import SQLEquivalenceInput, SQLEquivalenceOutput, SQLEquivalencePrompt

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class SQLSemanticEquivalence(BaseMetric):
    """
    Evaluates semantic equivalence between a generated SQL query and a reference query.

    This metric uses an LLM to analyze whether two SQL queries would produce the same
    results when executed against the same database, regardless of syntactic differences.
    The metric considers the database schema context to make accurate equivalence judgments.

    The metric returns:
    - 1.0 if the queries are semantically equivalent
    - 0.0 if the queries are not equivalent

    Usage:
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import SQLSemanticEquivalence
        >>>
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> metric = SQLSemanticEquivalence(llm=llm)
        >>>
        >>> result = await metric.ascore(
        ...     response="SELECT id, name FROM users WHERE active = true;",
        ...     reference="SELECT id, name FROM users WHERE active = 1;",
        ...     reference_contexts=[
        ...         "Table users: id (INT), name (VARCHAR), active (BOOLEAN)"
        ...     ],
        ... )
        >>> print(f"Equivalent: {result.value == 1.0}")

    Attributes:
        llm: Modern instructor-based LLM for SQL analysis
        name: The metric name (default: "sql_semantic_equivalence")
    """

    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "sql_semantic_equivalence",
        **kwargs,
    ):
        self.llm = llm
        self.equivalence_prompt = SQLEquivalencePrompt()
        super().__init__(name=name, **kwargs)

    async def ascore(
        self,
        response: str,
        reference: str,
        reference_contexts: Optional[List[str]] = None,
    ) -> MetricResult:
        """
        Calculate SQL semantic equivalence score.

        Args:
            response: The generated SQL query to evaluate
            reference: The reference SQL query to compare against
            reference_contexts: List of database schema descriptions providing
                context for the comparison. These are joined with newlines.

        Returns:
            MetricResult with equivalence score (1.0 if equivalent, 0.0 if not)
        """
        if not isinstance(response, str) or not response.strip():
            raise ValueError("response must be a non-empty SQL query string")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("reference must be a non-empty SQL query string")

        database_schema = ""
        if reference_contexts:
            database_schema = "\n".join(reference_contexts)

        input_data = SQLEquivalenceInput(
            reference=reference,
            response=response,
            database_schema=database_schema,
        )

        prompt_str = self.equivalence_prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_str, SQLEquivalenceOutput)

        score = 1.0 if result.equivalent else 0.0

        return MetricResult(
            value=score,
            reason=f"Response: {result.response_explanation}\nReference: {result.reference_explanation}",
        )
