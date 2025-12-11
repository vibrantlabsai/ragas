"""Tests for DataCompyScore metric (collections implementation)."""

import math

import pytest

# Skip all tests in this module if datacompy.core.Compare is not available
# datacompy >= 0.14 moved Compare to datacompy.core
try:
    from datacompy.core import Compare  # noqa: F401
except ImportError:
    try:
        from datacompy import Compare  # noqa: F401
    except ImportError:
        pytest.skip(
            "datacompy with Compare class not available", allow_module_level=True
        )

from ragas.metrics.collections import DataCompyScore


class TestDataCompyScoreCollections:
    """Test cases for DataCompyScore metric from collections."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        metric = DataCompyScore()
        assert metric.name == "data_compare_score"
        assert metric.mode == "rows"
        assert metric.metric == "f1"

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        metric = DataCompyScore(mode="columns", metric="precision", name="custom_score")
        assert metric.name == "custom_score"
        assert metric.mode == "columns"
        assert metric.metric == "precision"

    def test_init_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be either 'rows' or 'columns'"):
            DataCompyScore(mode="invalid")

    def test_init_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(
            ValueError, match="metric must be either 'precision', 'recall', or 'f1'"
        ):
            DataCompyScore(metric="invalid")

    @pytest.mark.asyncio
    async def test_perfect_match_rows(self):
        """Test perfect match scenario with row comparison."""
        metric = DataCompyScore(mode="rows", metric="f1")

        reference = "id,name\n1,Alice\n2,Bob"
        response = "id,name\n1,Alice\n2,Bob"

        result = await metric.ascore(reference=reference, response=response)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_partial_match_rows_f1(self):
        """Test partial match with row comparison returning F1 score."""
        metric = DataCompyScore(mode="rows", metric="f1")

        reference = "id,name\n1,Alice\n2,Bob"
        response = "id,name\n1,Alice\n2,Bob\n3,Charlie"

        result = await metric.ascore(reference=reference, response=response)
        # 2 matching rows, 2 reference rows, 3 response rows
        # recall = 2/2 = 1.0, precision = 2/3 = 0.667
        # F1 = 2 * (1.0 * 0.667) / (1.0 + 0.667) = 0.8
        assert 0.79 <= result.value <= 0.81

    @pytest.mark.asyncio
    async def test_precision_mode(self):
        """Test precision metric calculation."""
        metric = DataCompyScore(mode="rows", metric="precision")

        reference = "id,name\n1,Alice\n2,Bob"
        response = "id,name\n1,Alice\n2,Bob\n3,Charlie"

        result = await metric.ascore(reference=reference, response=response)
        # precision = 2/3 = 0.667
        assert 0.66 <= result.value <= 0.67

    @pytest.mark.asyncio
    async def test_recall_mode(self):
        """Test recall metric calculation."""
        metric = DataCompyScore(mode="rows", metric="recall")

        reference = "id,name\n1,Alice\n2,Bob"
        response = "id,name\n1,Alice\n2,Bob\n3,Charlie"

        result = await metric.ascore(reference=reference, response=response)
        # recall = 2/2 = 1.0
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_columns_mode(self):
        """Test column comparison mode."""
        metric = DataCompyScore(mode="columns", metric="f1")

        reference = "id,name,age\n1,Alice,30\n2,Bob,25"
        response = "id,name,age\n1,Alice,30\n2,Bob,25"

        result = await metric.ascore(reference=reference, response=response)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_columns_mode_partial_match(self):
        """Test column comparison mode with partial match."""
        metric = DataCompyScore(mode="columns", metric="f1")

        reference = "id,name,age\n1,Alice,30\n2,Bob,25"
        response = "id,name,age\n1,Alice,31\n2,Bob,26"

        result = await metric.ascore(reference=reference, response=response)
        # id and name match (age doesn't), so 2/3 columns match
        # precision = 2/3, recall = 2/3, F1 = 2/3
        assert 0.66 <= result.value <= 0.67

    @pytest.mark.asyncio
    async def test_invalid_reference_type(self):
        """Test that non-string reference raises ValueError."""
        metric = DataCompyScore()

        with pytest.raises(ValueError, match="reference must be a CSV string"):
            await metric.ascore(reference=123, response="id\n1")

    @pytest.mark.asyncio
    async def test_invalid_response_type(self):
        """Test that non-string response raises ValueError."""
        metric = DataCompyScore()

        with pytest.raises(ValueError, match="response must be a CSV string"):
            await metric.ascore(reference="id\n1", response=123)

    @pytest.mark.asyncio
    async def test_no_matching_rows(self):
        """Test scenario with no matching rows."""
        metric = DataCompyScore(mode="rows", metric="f1")

        reference = "id,name\n1,Alice\n2,Bob"
        response = "id,name\n3,Charlie\n4,David"

        result = await metric.ascore(reference=reference, response=response)
        # No matching rows: precision=0, recall=0, F1=0
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_result_reason_contains_info(self):
        """Test that result reason contains mode and precision/recall info."""
        metric = DataCompyScore(mode="rows", metric="f1")

        reference = "id,name\n1,Alice\n2,Bob"
        response = "id,name\n1,Alice\n2,Bob"

        result = await metric.ascore(reference=reference, response=response)
        assert "Mode: rows" in result.reason
        assert "Precision:" in result.reason
        assert "Recall:" in result.reason

    @pytest.mark.asyncio
    async def test_empty_dataframes(self):
        """Test behavior with empty dataframes."""
        metric = DataCompyScore(mode="rows", metric="f1")

        reference = "id,name"
        response = "id,name"

        result = await metric.ascore(reference=reference, response=response)
        # Empty dataframes: 0 rows, so division by zero protection should kick in
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_csv_parse_error_returns_nan(self):
        """Test that CSV parsing errors return NaN with reason."""
        metric = DataCompyScore()

        # This is truly invalid CSV - unclosed quotes and binary-like data
        reference = '"unclosed\x00binary'
        response = "id\n1"

        result = await metric.ascore(reference=reference, response=response)
        # Parsing should fail or comparison should fail
        assert math.isnan(result.value) or result.value == 0.0

    def test_sync_score_method(self):
        """Test synchronous score method."""
        metric = DataCompyScore(mode="rows", metric="f1")

        reference = "id,name\n1,Alice\n2,Bob"
        response = "id,name\n1,Alice\n2,Bob"

        result = metric.score(reference=reference, response=response)
        assert result.value == 1.0
