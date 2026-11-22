from __future__ import annotations

import logging
import re
import typing as t

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

logger = logging.getLogger(__name__)


def extract_identifiers(sql: str) -> t.Set[str]:
    """Extract all table and column identifiers from a SQL query."""
    sql = re.sub(r"'[^']*'", "", sql)
    sql = re.sub(r'"[^"]*"', "", sql)
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", sql)
    sql_keywords = {
        "select", "from", "where", "and", "or", "not", "in", "is", "null",
        "join", "left", "right", "inner", "outer", "full", "on", "as",
        "group", "by", "order", "having", "limit", "offset", "distinct",
        "count", "sum", "avg", "min", "max", "case", "when", "then", "else",
        "end", "insert", "update", "delete", "into", "values", "set",
        "create", "drop", "alter", "table", "index", "view", "with", "union",
        "all", "exists", "between", "like", "asc", "desc", "true", "false",
    }
    return {t.lower() for t in tokens if t.lower() not in sql_keywords and len(t) > 1}


def extract_schema_identifiers(schema: str) -> t.Set[str]:
    """Extract all valid table and column names from a schema definition."""
    tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", schema)
    type_keywords = {
        "int", "integer", "varchar", "text", "boolean", "bool", "float",
        "double", "decimal", "date", "datetime", "timestamp", "char",
        "bigint", "smallint", "numeric", "real", "table", "column",
        "primary", "key", "foreign", "references", "not", "null",
        "unique", "default", "index", "type", "string", "number",
    }
    return {t.lower() for t in tokens if t.lower() not in type_keywords}


class SQLSchemaHallucination(BaseMetric):
    """
    Measures whether a generated SQL query references only valid tables
    and columns that exist in the provided database schema.

    A score of 1.0 means all identifiers in the SQL exist in the schema.
    A score of 0.0 means none of the identifiers exist in the schema.

    This is a non-LLM metric — it uses rule-based identifier extraction
    and is fast, deterministic, and requires no API calls.

    Usage:
        >>> from ragas.metrics.collections import SQLSchemaHallucination
        >>>
        >>> metric = SQLSchemaHallucination()
        >>>
        >>> result = await metric.ascore(
        ...     response="SELECT username FROM users WHERE user_id = 1",
        ...     reference_contexts=["Table users: - user_id: INT - username: VARCHAR"]
        ... )
        >>> print(f"Score: {result.value}")

    Attributes:
        name: The metric name
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(
        self,
        name: str = "sql_schema_hallucination",
        **kwargs,
    ):
        super().__init__(name=name, allowed_values=(0.0, 1.0), **kwargs)

    async def ascore(
        self,
        response: str,
        reference_contexts: t.List[str],
    ) -> MetricResult:
        """
        Calculate SQL schema hallucination score.

        Args:
            response: The generated SQL query to evaluate
            reference_contexts: List of schema definition strings

        Returns:
            MetricResult with score between 0.0 and 1.0
        """
        assert isinstance(response, str), "response must be a string"
        assert isinstance(reference_contexts, list), (
            "reference_contexts must be a list of schema strings"
        )

        if not response.strip():
            return MetricResult(value=0.0, reason="Empty SQL response")

        schema_text = "\n".join(reference_contexts)
        if not schema_text.strip():
            return MetricResult(value=0.0, reason="Empty schema")

        sql_identifiers = extract_identifiers(response)
        schema_identifiers = extract_schema_identifiers(schema_text)

        if not sql_identifiers:
            return MetricResult(
                value=1.0, reason="No identifiers found in SQL"
            )

        valid = sql_identifiers & schema_identifiers
        score = round(len(valid) / len(sql_identifiers), 4)

        hallucinated = sql_identifiers - schema_identifiers
        reason = (
            f"{len(valid)}/{len(sql_identifiers)} identifiers valid. "
            f"Hallucinated: {hallucinated if hallucinated else 'none'}"
        )

        return MetricResult(value=score, reason=reason)