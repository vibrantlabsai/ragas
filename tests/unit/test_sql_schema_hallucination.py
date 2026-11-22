import pytest
from ragas.metrics.collections import SQLSchemaHallucination


SCHEMA = [
    """
    Table users:
    - user_id: INT
    - username: VARCHAR
    - email: VARCHAR
    - created_at: TIMESTAMP

    Table orders:
    - order_id: INT
    - user_id: INT
    - total_amount: DECIMAL
    - status: VARCHAR
    - created_at: TIMESTAMP
    """
]


@pytest.fixture
def metric():
    return SQLSchemaHallucination()


@pytest.mark.asyncio
async def test_perfect_score_valid_sql(metric):
    """SQL referencing only valid schema identifiers should score 1.0."""
    result = await metric.ascore(
        response="SELECT username, email FROM users WHERE user_id = 1",
        reference_contexts=SCHEMA,
    )
    assert result.value == 1.0


@pytest.mark.asyncio
async def test_hallucinated_column(metric):
    """SQL referencing a non-existent column should score less than 1.0."""
    result = await metric.ascore(
        response="SELECT username, phone_number FROM users WHERE user_id = 1",
        reference_contexts=SCHEMA,
    )
    assert result.value < 1.0


@pytest.mark.asyncio
async def test_hallucinated_table(metric):
    """SQL referencing a non-existent table should score less than 1.0."""
    result = await metric.ascore(
        response="SELECT * FROM transactions WHERE user_id = 1",
        reference_contexts=SCHEMA,
    )
    assert result.value < 1.0


@pytest.mark.asyncio
async def test_fully_hallucinated_sql(metric):
    """SQL with no valid schema references should score 0.0."""
    result = await metric.ascore(
        response="SELECT fake_col FROM fake_table WHERE ghost_id = 1",
        reference_contexts=SCHEMA,
    )
    assert result.value == 0.0


@pytest.mark.asyncio
async def test_empty_response(metric):
    """Empty SQL should return 0.0."""
    result = await metric.ascore(
        response="",
        reference_contexts=SCHEMA,
    )
    assert result.value == 0.0


@pytest.mark.asyncio
async def test_empty_schema(metric):
    """Empty schema should return 0.0."""
    result = await metric.ascore(
        response="SELECT username FROM users",
        reference_contexts=[""],
    )
    assert result.value == 0.0


@pytest.mark.asyncio
async def test_join_query(metric):
    """Valid JOIN query across two tables should score 1.0."""
    result = await metric.ascore(
        response="""
            SELECT u.username, o.total_amount
            FROM users u
            JOIN orders o ON u.user_id = o.user_id
            WHERE o.status = 'completed'
        """,
        reference_contexts=SCHEMA,
    )
    assert result.value == 1.0


@pytest.mark.asyncio
async def test_score_is_between_0_and_1(metric):
    """Score should always be between 0.0 and 1.0."""
    result = await metric.ascore(
        response="SELECT username, ghost_column FROM users",
        reference_contexts=SCHEMA,
    )
    assert 0.0 <= result.value <= 1.0


@pytest.mark.asyncio
async def test_result_has_reason(metric):
    """MetricResult should include a reason string."""
    result = await metric.ascore(
        response="SELECT username FROM users",
        reference_contexts=SCHEMA,
    )
    assert result.reason is not None
    assert isinstance(result.reason, str)


def test_metric_name(metric):
    """Metric should have correct name."""
    assert metric.name == "sql_schema_hallucination"
