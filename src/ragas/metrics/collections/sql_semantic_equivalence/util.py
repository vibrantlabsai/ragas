"""SQLSemanticEquivalence prompt classes and models."""

import typing as t

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt


class SQLEquivalenceInput(BaseModel):
    reference: str = Field(..., description="Reference SQL query")
    response: str = Field(..., description="Generated SQL query to evaluate")
    database_schema: str = Field(..., description="Database schema for context")


class SQLEquivalenceOutput(BaseModel):
    response_explanation: str = Field(
        ..., description="Explanation of what the generated SQL query does"
    )
    reference_explanation: str = Field(
        ..., description="Explanation of what the reference SQL query does"
    )
    equivalent: bool = Field(
        ..., description="Whether the queries are semantically equivalent"
    )


class SQLEquivalencePrompt(BasePrompt[SQLEquivalenceInput, SQLEquivalenceOutput]):
    """Prompt for evaluating semantic equivalence between SQL queries."""

    input_model = SQLEquivalenceInput
    output_model = SQLEquivalenceOutput

    instruction = """Explain and compare two SQL queries (Q1 and Q2) based on the provided database schema. First, explain each query, then determine if they are semantically equivalent.

Two SQL queries are semantically equivalent if they would return the same results when executed against the same database, regardless of syntactic differences like:
- Different but equivalent boolean expressions (1 vs true)
- Column ordering in SELECT (when not affecting results)
- Alias naming differences
- Whitespace and formatting"""

    examples: t.List[t.Tuple[SQLEquivalenceInput, SQLEquivalenceOutput]] = [
        (
            SQLEquivalenceInput(
                reference="SELECT id, name FROM users WHERE active = 1;",
                response="SELECT id, name FROM users WHERE active = true;",
                database_schema="""Table users:
- id: INT
- name: VARCHAR
- active: BOOLEAN""",
            ),
            SQLEquivalenceOutput(
                response_explanation="The generated SQL query retrieves the id and name of users where the active field is true.",
                reference_explanation="The reference SQL query retrieves the id and name of users where the active field equals 1.",
                equivalent=True,
            ),
        ),
        (
            SQLEquivalenceInput(
                reference="SELECT product_name, SUM(quantity) AS total FROM orders GROUP BY product_name;",
                response="SELECT product_name, COUNT(quantity) AS total FROM orders GROUP BY product_name;",
                database_schema="""Table orders:
- order_id: INT
- product_name: VARCHAR
- quantity: INT""",
            ),
            SQLEquivalenceOutput(
                response_explanation="The generated SQL query retrieves product names with a COUNT of their quantities, which counts the number of non-null quantity values.",
                reference_explanation="The reference SQL query retrieves product names with a SUM of their quantities, which adds up all quantity values.",
                equivalent=False,
            ),
        ),
    ]
