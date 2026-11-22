"""SQLSchemaHallucination metric - detects hallucinated table/column references."""
from ragas.metrics.collections.sql_schema_hallucination.metric import (
    SQLSchemaHallucination,
)

__all__ = ["SQLSchemaHallucination"]