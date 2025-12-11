"""DataCompyScore metric - Modern collections implementation."""

import logging
import typing as t
from io import StringIO

import numpy as np

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

logger = logging.getLogger(__name__)


class DataCompyScore(BaseMetric):
    """
    Compare CSV data using datacompy library to compute precision, recall, or F1 scores.

    This metric compares two CSV strings (reference and response) and calculates
    matching statistics at either row or column level. Useful for evaluating
    SQL-to-text or data generation tasks where tabular output needs to be compared.

    The metric supports three modes of comparison:
    - precision: Proportion of response rows/columns that match reference
    - recall: Proportion of reference rows/columns found in response
    - f1: Harmonic mean of precision and recall

    Usage:
        >>> from ragas.metrics.collections import DataCompyScore
        >>>
        >>> metric = DataCompyScore(mode="rows", metric="f1")
        >>>
        >>> result = await metric.ascore(
        ...     reference="id,name\\n1,Alice\\n2,Bob",
        ...     response="id,name\\n1,Alice\\n2,Bob\\n3,Charlie",
        ... )
        >>> print(f"F1 Score: {result.value}")

    Attributes:
        name: The metric name (default: "data_compare_score")
        mode: Comparison mode - "rows" or "columns"
        metric: Score type - "precision", "recall", or "f1"
    """

    def __init__(
        self,
        mode: t.Literal["rows", "columns"] = "rows",
        metric: t.Literal["precision", "recall", "f1"] = "f1",
        name: str = "data_compare_score",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Check for required dependencies at init time
        try:
            import pandas as pd

            # Try new import path first (datacompy >= 0.14), fall back to legacy
            try:
                from datacompy.core import Compare
            except ImportError:
                from datacompy import Compare  # type: ignore[attr-defined]
        except ImportError as e:
            raise ImportError(
                f"{e.name} is required for DataCompyScore. "
                f"Please install it using `pip install {e.name}`"
            )

        self._pd = pd
        self._Compare = Compare

        if mode not in ["rows", "columns"]:
            raise ValueError("mode must be either 'rows' or 'columns'")
        if metric not in ["precision", "recall", "f1"]:
            raise ValueError("metric must be either 'precision', 'recall', or 'f1'")

        self.mode = mode
        self.metric = metric

    async def ascore(
        self,
        reference: str,
        response: str,
    ) -> MetricResult:
        """
        Calculate data comparison score between reference and response CSV strings.

        Args:
            reference: The reference CSV data as a string
            response: The response CSV data to evaluate

        Returns:
            MetricResult with comparison score (0.0-1.0) or NaN if parsing fails
        """
        if not isinstance(reference, str):
            raise ValueError("reference must be a CSV string")
        if not isinstance(response, str):
            raise ValueError("response must be a CSV string")

        try:
            reference_df = self._pd.read_csv(StringIO(reference))
            response_df = self._pd.read_csv(StringIO(response))
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return MetricResult(value=float(np.nan), reason=f"CSV parsing error: {e}")

        compare = self._Compare(reference_df, response_df, on_index=True)

        if self.mode == "rows":
            matching_rows = compare.count_matching_rows()
            recall = (
                matching_rows / reference_df.shape[0]
                if reference_df.shape[0] > 0
                else 0.0
            )
            precision = (
                matching_rows / response_df.shape[0]
                if response_df.shape[0] > 0
                else 0.0
            )
        else:
            matched_cols = len(
                [col for col in compare.column_stats if col["unequal_cnt"] == 0]
            )
            recall = (
                matched_cols / reference_df.shape[1]
                if reference_df.shape[1] > 0
                else 0.0
            )
            precision = (
                matched_cols / response_df.shape[1] if response_df.shape[1] > 0 else 0.0
            )

        if self.metric == "precision":
            score = precision
        elif self.metric == "recall":
            score = recall
        else:
            if precision + recall == 0:
                score = 0.0
            else:
                score = 2 * (precision * recall) / (precision + recall)

        return MetricResult(
            value=float(score),
            reason=f"Mode: {self.mode}, Precision: {precision:.4f}, Recall: {recall:.4f}",
        )
