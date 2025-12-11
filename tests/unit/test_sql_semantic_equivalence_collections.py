"""Tests for SQLSemanticEquivalence metric (collections implementation)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import SQLSemanticEquivalence
from ragas.metrics.collections.sql_semantic_equivalence.util import SQLEquivalenceOutput


class MockInstructorLLM(InstructorBaseRagasLLM):
    """Mock implementation of InstructorBaseRagasLLM for testing."""

    def __init__(self):
        self.agenerate = AsyncMock()
        self.generate = MagicMock()

    def generate(self, prompt, response_model):
        return self.generate(prompt, response_model)

    async def agenerate(self, prompt, response_model):
        return await self.agenerate(prompt, response_model)


@pytest.fixture
def mock_llm():
    """Fixture providing a mock LLM."""
    return MockInstructorLLM()


class TestSQLSemanticEquivalenceCollections:
    """Test cases for SQLSemanticEquivalence metric from collections."""

    @pytest.mark.asyncio
    async def test_equivalent_queries_boolean_syntax(self, mock_llm):
        """Test equivalent queries with different boolean syntax."""
        mock_llm.agenerate.return_value = SQLEquivalenceOutput(
            response_explanation="Query selects active users using boolean true",
            reference_explanation="Query selects active users using numeric 1",
            equivalent=True,
        )
        metric = SQLSemanticEquivalence(llm=mock_llm)

        result = await metric.ascore(
            response="SELECT id, name FROM users WHERE active = true;",
            reference="SELECT id, name FROM users WHERE active = 1;",
            reference_contexts=[
                "Table users: id (INT), name (VARCHAR), active (BOOLEAN)"
            ],
        )

        assert result.value == 1.0
        assert "response" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_non_equivalent_queries_sum_vs_count(self, mock_llm):
        """Test non-equivalent queries using SUM vs COUNT."""
        mock_llm.agenerate.return_value = SQLEquivalenceOutput(
            response_explanation="Query counts quantity values",
            reference_explanation="Query sums quantity values",
            equivalent=False,
        )
        metric = SQLSemanticEquivalence(llm=mock_llm)

        result = await metric.ascore(
            response="SELECT product_name, COUNT(quantity) FROM orders GROUP BY product_name;",
            reference="SELECT product_name, SUM(quantity) FROM orders GROUP BY product_name;",
            reference_contexts=[
                "Table orders: order_id (INT), product_name (VARCHAR), quantity (INT)"
            ],
        )

        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_equivalent_queries_with_join(self, mock_llm):
        """Test equivalent queries with JOIN operations."""
        mock_llm.agenerate.return_value = SQLEquivalenceOutput(
            response_explanation="Query joins order_items with products and sums quantities",
            reference_explanation="Query performs identical join and aggregation",
            equivalent=True,
        )
        metric = SQLSemanticEquivalence(llm=mock_llm)

        result = await metric.ascore(
            response="""
                SELECT p.product_name, SUM(oi.quantity) AS total_quantity
                FROM order_items oi
                JOIN products p ON oi.product_id = p.product_id
                GROUP BY p.product_name;
            """,
            reference="""
                SELECT products.product_name, SUM(order_items.quantity) AS total_quantity
                FROM order_items
                INNER JOIN products ON order_items.product_id = products.product_id
                GROUP BY products.product_name;
            """,
            reference_contexts=[
                """Table order_items:
                - order_item_id: INT
                - order_id: INT
                - product_id: INT
                - quantity: INT""",
                """Table products:
                - product_id: INT
                - product_name: VARCHAR
                - price: DECIMAL""",
            ],
        )

        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_empty_reference_contexts(self, mock_llm):
        """Test with empty reference contexts (no schema)."""
        mock_llm.agenerate.return_value = SQLEquivalenceOutput(
            response_explanation="Query selects all from users",
            reference_explanation="Query selects all from users",
            equivalent=True,
        )
        metric = SQLSemanticEquivalence(llm=mock_llm)

        result = await metric.ascore(
            response="SELECT * FROM users;",
            reference="SELECT * FROM users;",
            reference_contexts=[],
        )

        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_none_reference_contexts(self, mock_llm):
        """Test with None reference contexts."""
        mock_llm.agenerate.return_value = SQLEquivalenceOutput(
            response_explanation="Query selects all from users",
            reference_explanation="Query selects all from users",
            equivalent=True,
        )
        metric = SQLSemanticEquivalence(llm=mock_llm)

        result = await metric.ascore(
            response="SELECT * FROM users;",
            reference="SELECT * FROM users;",
            reference_contexts=None,
        )

        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_empty_response_raises_error(self, mock_llm):
        """Test that empty response raises ValueError."""
        metric = SQLSemanticEquivalence(llm=mock_llm)

        with pytest.raises(ValueError, match="response must be a non-empty"):
            await metric.ascore(
                response="",
                reference="SELECT * FROM users;",
            )

    @pytest.mark.asyncio
    async def test_empty_reference_raises_error(self, mock_llm):
        """Test that empty reference raises ValueError."""
        metric = SQLSemanticEquivalence(llm=mock_llm)

        with pytest.raises(ValueError, match="reference must be a non-empty"):
            await metric.ascore(
                response="SELECT * FROM users;",
                reference="",
            )

    @pytest.mark.asyncio
    async def test_whitespace_only_response_raises_error(self, mock_llm):
        """Test that whitespace-only response raises ValueError."""
        metric = SQLSemanticEquivalence(llm=mock_llm)

        with pytest.raises(ValueError, match="response must be a non-empty"):
            await metric.ascore(
                response="   ",
                reference="SELECT * FROM users;",
            )

    @pytest.mark.asyncio
    async def test_multiple_schema_contexts_joined(self, mock_llm):
        """Test that multiple schema contexts are properly joined."""
        mock_llm.agenerate.return_value = SQLEquivalenceOutput(
            response_explanation="test",
            reference_explanation="test",
            equivalent=True,
        )
        metric = SQLSemanticEquivalence(llm=mock_llm)

        await metric.ascore(
            response="SELECT * FROM orders o JOIN products p ON o.product_id = p.id;",
            reference="SELECT * FROM orders o JOIN products p ON o.product_id = p.id;",
            reference_contexts=[
                "Table orders: id, product_id, quantity",
                "Table products: id, name, price",
            ],
        )

        # Verify both schema parts appear in the prompt
        call_args = mock_llm.agenerate.call_args
        prompt_str = call_args[0][0]
        assert "Table orders" in prompt_str
        assert "Table products" in prompt_str

    @pytest.mark.asyncio
    async def test_result_includes_explanations(self, mock_llm):
        """Test that result includes explanations from LLM."""
        mock_llm.agenerate.return_value = SQLEquivalenceOutput(
            response_explanation="The response query selects all users",
            reference_explanation="The reference query also selects all users",
            equivalent=True,
        )
        metric = SQLSemanticEquivalence(llm=mock_llm)

        result = await metric.ascore(
            response="SELECT * FROM users;",
            reference="SELECT * FROM users;",
        )

        assert "response query selects all users" in result.reason
        assert "reference query also selects all users" in result.reason

    @pytest.mark.asyncio
    async def test_custom_metric_name(self, mock_llm):
        """Test that custom metric name is applied."""
        metric = SQLSemanticEquivalence(llm=mock_llm, name="my_sql_metric")

        assert metric.name == "my_sql_metric"

    def test_sync_score_method(self, mock_llm):
        """Test synchronous score method."""
        mock_llm.agenerate.return_value = SQLEquivalenceOutput(
            response_explanation="test",
            reference_explanation="test",
            equivalent=True,
        )
        metric = SQLSemanticEquivalence(llm=mock_llm)

        result = metric.score(
            response="SELECT * FROM users;",
            reference="SELECT * FROM users;",
        )

        assert result.value == 1.0


class TestSQLEquivalencePrompt:
    """Test cases for SQLEquivalencePrompt."""

    def test_prompt_has_required_attributes(self):
        """Test that prompt class has all required attributes."""
        from ragas.metrics.collections.sql_semantic_equivalence.util import (
            SQLEquivalencePrompt,
        )

        prompt = SQLEquivalencePrompt()

        assert hasattr(prompt, "instruction")
        assert hasattr(prompt, "input_model")
        assert hasattr(prompt, "output_model")
        assert hasattr(prompt, "examples")
        assert len(prompt.examples) >= 1

    def test_prompt_to_string(self):
        """Test prompt generates valid string."""
        from ragas.metrics.collections.sql_semantic_equivalence.util import (
            SQLEquivalenceInput,
            SQLEquivalencePrompt,
        )

        prompt = SQLEquivalencePrompt()
        input_data = SQLEquivalenceInput(
            reference="SELECT * FROM users;",
            response="SELECT * FROM users;",
            database_schema="Table users: id, name",
        )

        prompt_str = prompt.to_string(input_data)

        assert "SELECT * FROM users" in prompt_str
        assert "Table users" in prompt_str
        assert "equivalent" in prompt_str.lower() or "EXAMPLES" in prompt_str

    def test_prompt_examples_cover_both_cases(self):
        """Test that prompt examples cover both equivalent and non-equivalent cases."""
        from ragas.metrics.collections.sql_semantic_equivalence.util import (
            SQLEquivalencePrompt,
        )

        prompt = SQLEquivalencePrompt()

        equivalence_values = [ex[1].equivalent for ex in prompt.examples]
        assert True in equivalence_values, "Should have an example with equivalent=True"
        assert False in equivalence_values, (
            "Should have an example with equivalent=False"
        )
