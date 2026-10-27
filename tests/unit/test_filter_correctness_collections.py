"""Tests for FilterCorrectness metric (collections implementation)."""

import pytest

from ragas.metrics.collections.filter_correctness import FilterCorrectness


@pytest.fixture
def metric():
    """Fixture providing FilterCorrectness instance."""
    return FilterCorrectness()


class TestFilterCorrectnessCollections:
    """Test cases for FilterCorrectness metric from collections."""

    @pytest.mark.asyncio
    async def test_all_chunks_respect_equality_filter(self, metric):
        result = await metric.ascore(
            retrieved_metadatas=[
                {"tenant_id": "A", "doc": "1"},
                {"tenant_id": "A", "doc": "2"},
            ],
            filter_field="tenant_id",
            allowed_values="A",
        )
        assert result.value == 1.0
        assert result.traces is not None
        assert result.traces["output"]["violations"] == 0
        assert result.traces["output"]["checked"] == 2
        assert result.traces["output"]["status"] == "scored"

    @pytest.mark.asyncio
    async def test_leak_ratio_with_offending_values_in_reason(self, metric):
        result = await metric.ascore(
            retrieved_metadatas=[
                {"jurisdiction": "domestic"},
                {"jurisdiction": "international"},
                {"jurisdiction": "domestic"},
                {"jurisdiction": "international"},
                {"jurisdiction": "domestic"},
            ],
            filter_field="jurisdiction",
            allowed_values="domestic",
        )
        # 2 of 5 violate → correctness 0.6
        assert result.value == 0.6
        assert result.traces["output"]["violations"] == 2
        assert result.traces["output"]["checked"] == 5
        assert "international" in (result.reason or "")
        assert set(result.traces["output"]["offending_values"]) == {"international"}

    @pytest.mark.asyncio
    async def test_in_filter_allows_membership(self, metric):
        result = await metric.ascore(
            retrieved_metadatas=[
                {"category": "public"},
                {"category": "partner"},
            ],
            filter_field="category",
            allowed_values=["public", "partner"],
        )
        assert result.value == 1.0
        assert result.traces["output"]["violations"] == 0

    @pytest.mark.asyncio
    async def test_in_filter_detects_disallowed_value(self, metric):
        result = await metric.ascore(
            retrieved_metadatas=[
                {"category": "public"},
                {"category": "internal"},
            ],
            filter_field="category",
            allowed_values=["public", "partner"],
        )
        assert result.value == 0.5
        assert result.traces["output"]["violations"] == 1
        assert result.traces["output"]["offending_values"] == ["internal"]

    @pytest.mark.asyncio
    async def test_integer_equality_filter(self, metric):
        result = await metric.ascore(
            retrieved_metadatas=[
                {"tenant_id": 42},
                {"tenant_id": 7},
            ],
            filter_field="tenant_id",
            allowed_values=42,
        )
        assert result.value == 0.5
        assert result.traces["output"]["offending_values"] == [7]

    @pytest.mark.asyncio
    async def test_empty_retrieval_is_not_perfect_score(self, metric):
        result = await metric.ascore(
            retrieved_metadatas=[],
            filter_field="tenant_id",
            allowed_values="A",
        )
        assert result.value == 0.0
        assert result.traces["output"]["status"] == "empty_retrieval"
        assert "empty" in (result.reason or "").lower()

    @pytest.mark.asyncio
    async def test_missing_metadata_separated_from_violations(self, metric):
        result = await metric.ascore(
            retrieved_metadatas=[
                {"tenant_id": "A"},
                {"other": "x"},  # missing filter field
                {"tenant_id": None},  # null treated as missing
                {"tenant_id": "B"},  # violation
            ],
            filter_field="tenant_id",
            allowed_values="A",
        )
        # checked = 2 (A and B); violations = 1 (B); missing = 2
        assert result.value == 0.5
        assert result.traces["output"]["checked"] == 2
        assert result.traces["output"]["violations"] == 1
        assert result.traces["output"]["missing_metadata"] == 2
        assert result.traces["output"]["offending_values"] == ["B"]

    @pytest.mark.asyncio
    async def test_all_missing_metadata_is_not_perfect_score(self, metric):
        result = await metric.ascore(
            retrieved_metadatas=[{"x": 1}, {"y": 2}],
            filter_field="tenant_id",
            allowed_values="A",
        )
        assert result.value == 0.0
        assert result.traces["output"]["status"] == "missing_metadata"
        assert result.traces["output"]["checked"] == 0
        assert result.traces["output"]["missing_metadata"] == 2

    @pytest.mark.asyncio
    async def test_rejects_non_list_metadatas(self, metric):
        with pytest.raises(ValueError, match="retrieved_metadatas"):
            await metric.ascore(
                retrieved_metadatas="not-a-list",  # type: ignore[arg-type]
                filter_field="tenant_id",
                allowed_values="A",
            )

    @pytest.mark.asyncio
    async def test_rejects_empty_filter_field(self, metric):
        with pytest.raises(ValueError, match="filter_field"):
            await metric.ascore(
                retrieved_metadatas=[{"tenant_id": "A"}],
                filter_field="   ",
                allowed_values="A",
            )

    @pytest.mark.asyncio
    async def test_rejects_empty_allowed_values(self, metric):
        with pytest.raises(ValueError, match="allowed_values"):
            await metric.ascore(
                retrieved_metadatas=[{"tenant_id": "A"}],
                filter_field="tenant_id",
                allowed_values=[],
            )

    @pytest.mark.asyncio
    async def test_list_valued_metadata_must_be_subset_of_allowed(self, metric):
        result = await metric.ascore(
            retrieved_metadatas=[
                {"tags": ["public", "partner"]},
                {"tags": ["public", "secret"]},
            ],
            filter_field="tags",
            allowed_values=["public", "partner"],
        )
        assert result.value == 0.5
        assert result.traces["output"]["violations"] == 1
        assert "secret" in result.traces["output"]["offending_values"]

    def test_sync_score_wrapper(self, metric):
        result = metric.score(
            retrieved_metadatas=[{"tenant_id": "A"}],
            filter_field="tenant_id",
            allowed_values="A",
        )
        assert result.value == 1.0
        assert result.traces["output"]["status"] == "scored"
