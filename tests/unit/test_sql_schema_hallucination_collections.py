"""Tests for SQLSchemaHallucination metric (collections implementation)."""

import pytest

from ragas.metrics.collections import SQLSchemaHallucination
from ragas.metrics.collections.sql_schema_hallucination.util import (
    extract_references,
    score_references,
)

SCHEMA = {
    "users": ["id", "name", "email"],
    "orders": ["id", "user_id", "total"],
}


class TestSQLReferenceExtraction:
    """Test cases for the SQL parsing utilities."""

    def test_extracts_table_and_columns(self):
        refs = extract_references("SELECT id, name FROM users")
        assert refs.tables == {"users"}
        assert refs.columns == {(None, "id"), (None, "name")}

    def test_wildcard_is_ignored(self):
        refs = extract_references("SELECT * FROM users")
        assert refs.columns == set()
        assert refs.tables == {"users"}

    def test_function_name_is_not_a_column(self):
        refs = extract_references("SELECT COUNT(*) FROM users")
        assert refs.columns == set()

    def test_column_alias_is_excluded(self):
        refs = extract_references("SELECT id AS user_id FROM users")
        assert refs.columns == {(None, "id")}

    def test_table_alias_resolves_qualified_column(self):
        refs = extract_references("SELECT u.email FROM users u")
        assert refs.columns == {("users", "email")}

    def test_cte_name_is_not_a_table(self):
        refs = extract_references("WITH t AS (SELECT id FROM users) SELECT id FROM t")
        assert "t" not in refs.tables
        assert "users" in refs.tables

    def test_where_clause_columns_are_extracted(self):
        refs = extract_references("SELECT name FROM users WHERE emial = 'x'")
        assert (None, "emial") in refs.columns

    def test_empty_query_yields_no_references(self):
        refs = extract_references("")
        assert refs.tables == set()
        assert refs.columns == set()

    def test_score_counts_valid_and_hallucinated(self):
        refs = extract_references("SELECT emial FROM users")
        valid, total, hallucinated = score_references(refs, SCHEMA)
        assert (valid, total) == (1, 2)
        assert hallucinated == ["emial"]

    def test_column_inside_function_is_extracted(self):
        refs = extract_references("SELECT LOWER(emial) FROM users")
        assert (None, "emial") in refs.columns

    def test_derived_table_alias_is_not_scored(self):
        refs = extract_references("SELECT x.id FROM (SELECT id FROM users) x")
        assert refs.tables == {"users"}
        assert ("x", "id") not in refs.columns

    def test_cte_qualified_column_is_not_scored(self):
        refs = extract_references(
            "WITH recent AS (SELECT id FROM users) SELECT recent.id FROM recent"
        )
        assert ("recent", "id") not in refs.columns

    def test_multiple_statements_are_all_extracted(self):
        refs = extract_references("SELECT id FROM users; SELECT emial FROM users")
        assert (None, "emial") in refs.columns

    def test_update_target_is_a_table(self):
        refs = extract_references("UPDATE users SET email = 'x' WHERE id = 1")
        assert refs.tables == {"users"}
        assert (None, "users") not in refs.columns


class TestSQLSchemaHallucination:
    """Test cases for the SQLSchemaHallucination metric."""

    def test_init_default_name(self):
        metric = SQLSchemaHallucination()
        assert metric.name == "sql_schema_hallucination"

    def test_init_custom_name(self):
        metric = SQLSchemaHallucination(name="custom")
        assert metric.name == "custom"

    @pytest.mark.asyncio
    async def test_all_references_valid(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(
            response="SELECT id, name FROM users", schema=SCHEMA
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_hallucinated_column(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(response="SELECT emial FROM users", schema=SCHEMA)
        assert result.value == 0.5
        assert "emial" in result.reason

    @pytest.mark.asyncio
    async def test_hallucinated_table(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(response="SELECT id FROM userz", schema=SCHEMA)
        assert result.value == 0.5
        assert "userz" in result.reason

    @pytest.mark.asyncio
    async def test_qualified_column_via_alias(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(
            response="SELECT o.total FROM orders o JOIN users u ON o.user_id = u.id",
            schema=SCHEMA,
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_select_star_scores_one(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(response="SELECT * FROM users", schema=SCHEMA)
        assert result.value == 1.0
        assert result.reason == "Valid references 1/1"

    @pytest.mark.asyncio
    async def test_cte_name_not_flagged(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(
            response="WITH recent AS (SELECT id FROM users) SELECT id FROM recent",
            schema=SCHEMA,
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(
            response="SELECT ID, NAME FROM USERS", schema=SCHEMA
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_empty_response_raises(self):
        metric = SQLSchemaHallucination()
        with pytest.raises(ValueError, match="non-empty SQL query"):
            await metric.ascore(response="  ", schema=SCHEMA)

    @pytest.mark.asyncio
    async def test_empty_schema_raises(self):
        metric = SQLSchemaHallucination()
        with pytest.raises(ValueError, match="non-empty mapping"):
            await metric.ascore(response="SELECT id FROM users", schema={})

    @pytest.mark.asyncio
    async def test_hallucinated_column_inside_function(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(
            response="SELECT LOWER(emial) FROM users", schema=SCHEMA
        )
        assert result.value == 0.5
        assert "emial" in result.reason

    @pytest.mark.asyncio
    async def test_derived_table_not_flagged(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(
            response="SELECT x.id FROM (SELECT id FROM users) x", schema=SCHEMA
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_unqualified_column_scoped_to_referenced_tables(self):
        metric = SQLSchemaHallucination()
        # 'total' exists in 'orders' but the query only references 'users'.
        result = await metric.ascore(response="SELECT total FROM users", schema=SCHEMA)
        assert result.value == 0.5
        assert "total" in result.reason

    @pytest.mark.asyncio
    async def test_subquery_tables_do_not_validate_outer_columns(self):
        metric = SQLSchemaHallucination()
        # 'total' is in 'orders' (used only in the subquery), not in 'users'.
        result = await metric.ascore(
            response="SELECT total FROM users WHERE id IN (SELECT user_id FROM orders)",
            schema=SCHEMA,
        )
        assert result.value < 1.0
        assert "total" in result.reason

    @pytest.mark.asyncio
    async def test_correlated_subquery_resolves_outer_alias(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(
            response=(
                "SELECT id FROM orders o "
                "WHERE EXISTS (SELECT 1 FROM users u WHERE u.id = o.user_id)"
            ),
            schema=SCHEMA,
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_valid_subquery_does_not_false_flag(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(
            response="SELECT name FROM users WHERE id IN (SELECT user_id FROM orders)",
            schema=SCHEMA,
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_second_statement_is_scored(self):
        metric = SQLSchemaHallucination()
        result = await metric.ascore(
            response="SELECT id FROM users; SELECT emial FROM users", schema=SCHEMA
        )
        assert result.value < 1.0
        assert "emial" in result.reason

    def test_sync_score_method(self):
        metric = SQLSchemaHallucination()
        result = metric.score(response="SELECT emial FROM users", schema=SCHEMA)
        assert result.value == 0.5
