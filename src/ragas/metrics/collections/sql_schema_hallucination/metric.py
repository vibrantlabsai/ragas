"""SQLSchemaHallucination metric - Modern collections implementation."""

import typing as t

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import Schema, extract_references, score_references


class SQLSchemaHallucination(BaseMetric):
    """
    Measure how much a generated SQL query references a schema that exists.

    Text-to-SQL models often invent plausible but non-existent tables and
    columns, especially over wide schemas. This non-LLM metric parses the query,
    collects the tables and columns it references, and checks them against a
    provided schema. The score is the fraction of references that are valid, so
    1.0 means no hallucinated references and lower values mean the query names
    tables or columns the schema does not contain.

    ``SELECT *`` and function calls claim no specific column and are ignored.
    In-query aliases and CTE names are resolved and never counted as
    hallucinations. A query that references nothing (e.g. ``SELECT *``) scores
    1.0.

    Usage:
        >>> from ragas.metrics.collections import SQLSchemaHallucination
        >>>
        >>> metric = SQLSchemaHallucination()
        >>>
        >>> result = await metric.ascore(
        ...     response="SELECT emial FROM users",
        ...     schema={"users": ["id", "name", "email"]},
        ... )
        >>> print(result.value)  # 0.5 - 'emial' is hallucinated
        >>> print(result.reason)

    Attributes:
        name: The metric name (default: "sql_schema_hallucination")
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(
        self,
        name: str = "sql_schema_hallucination",
        **base_kwargs,
    ):
        super().__init__(name=name, **base_kwargs)

        try:
            import sqlparse  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"{e.name} is required for SQLSchemaHallucination. "
                f"Please install it using `pip install {e.name}`"
            )

    async def ascore(
        self,
        response: str,
        schema: Schema,
    ) -> MetricResult:
        """
        Calculate the schema hallucination score for a generated SQL query.

        Args:
            response: The generated SQL query to evaluate.
            schema: Mapping of table name to its column names, e.g.
                ``{"users": ["id", "name"], "orders": ["id", "user_id"]}``.

        Returns:
            MetricResult with the fraction of valid references (0.0-1.0). The
            reason lists any hallucinated references.
        """
        if not isinstance(response, str) or not response.strip():
            raise ValueError("response must be a non-empty SQL query string")
        if not isinstance(schema, t.Mapping) or not schema:
            raise ValueError("schema must be a non-empty mapping of table to columns")

        references = extract_references(response)
        valid, total, hallucinated = score_references(references, schema)

        if total == 0:
            return MetricResult(
                value=1.0,
                reason="No table or column references to validate",
            )

        score = valid / total
        reason = f"Valid references {valid}/{total}"
        if hallucinated:
            reason += f"; hallucinated: {', '.join(sorted(hallucinated))}"

        return MetricResult(value=float(score), reason=reason)
