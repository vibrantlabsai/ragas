from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import Metric, MetricType, SingleTurnMetric
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from datasets import Dataset
    from langchain_core.callbacks import Callbacks


@dataclass
class _RiskControlCalculator:
    """
    A private helper class to perform the dataset-wide calculations for the risk control suite.
    This class is instantiated once and shared across all four metrics to ensure the calculation
    is performed only once.
    """

    dataset: Dataset
    _scores: dict[str, float] | None = field(default=None, init=False, repr=False)

    def _calculate(self) -> None:
        """
        Iterates through the dataset to count the four outcomes (AK, UK, AD, UD) and
        computes the four risk-control metrics.
        """
        required_columns = {"ground_truth_answerable", "model_decision"}
        for col in required_columns:
            if col not in self.dataset.column_names:
                raise ValueError(
                    f"Missing required column '{col}' in the dataset for Risk-Control metrics. "
                    "Please ensure your dataset contains 'ground_truth_answerable' (boolean) and 'model_decision' ('kept'/'discarded') columns."
                )

        # The four outcomes
        ak_count, uk_count, ad_count, ud_count = 0, 0, 0, 0

        for row in self.dataset:
            is_answerable = row["ground_truth_answerable"]
            decision_is_kept = row["model_decision"].lower() == "kept"

            if is_answerable and decision_is_kept:
                ak_count += 1
            elif not is_answerable and decision_is_kept:
                uk_count += 1
            elif is_answerable and not decision_is_kept:
                ad_count += 1
            elif not is_answerable and not decision_is_kept:
                ud_count += 1

        total_kept = ak_count + uk_count
        total_unanswerable = uk_count + ud_count
        total_decisions = ak_count + uk_count + ad_count + ud_count

        # Risk: Probability that a kept answer is risky. Lower is better.
        risk = uk_count / total_kept if total_kept > 0 else 0.0

        # Carefulness: Recall for the "unanswerable" class. Higher is better.
        carefulness = ud_count / total_unanswerable if total_unanswerable > 0 else 0.0

        # Alignment: Overall accuracy of the keep/discard decision. Higher is better.
        alignment = (ak_count + ud_count) / total_decisions if total_decisions > 0 else 0.0

        # Coverage: Proportion of questions the system attempts to answer. Higher is better.
        coverage = total_kept / total_decisions if total_decisions > 0 else 0.0

        self._scores = {
            "risk": risk,
            "carefulness": carefulness,
            "alignment": alignment,
            "coverage": coverage,
        }

    def get_scores(self) -> dict[str, float]:
        """
        Returns the calculated scores. If not already calculated, triggers the calculation.
        """
        if self._scores is None:
            self._calculate()
        assert self._scores is not None
        return self._scores


@dataclass(kw_only=True)
class Risk(SingleTurnMetric):
    """
    Measures the probability that an answer provided by the system is a "risky"
    one (i.e., it should have been discarded). A lower Risk score is better.
    """
    calculator: _RiskControlCalculator
    name: str = "risk"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(default_factory=dict)

    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks: Callbacks) -> float:
        return self.calculator.get_scores()["risk"]


@dataclass(kw_only=True)
class Carefulness(SingleTurnMetric):
    """
    Measures the system's ability to correctly identify and discard unanswerable
    questions. It is effectively the recall for the "unanswerable" class.
    """
    calculator: _RiskControlCalculator
    name: str = "carefulness"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(default_factory=dict)
    
    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks: Callbacks) -> float:
        return self.calculator.get_scores()["carefulness"]


@dataclass(kw_only=True)
class Alignment(SingleTurnMetric):
    """
    Measures the overall accuracy of the model's decision-making process
    (both its decisions to keep and to discard).
    """
    calculator: _RiskControlCalculator
    name: str = "alignment"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(default_factory=dict)

    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks: Callbacks) -> float:
        return self.calculator.get_scores()["alignment"]


@dataclass(kw_only=True)
class Coverage(SingleTurnMetric):
    """
    Measures the proportion of questions that the system attempts to answer.
    It quantifies the system's "helpfulness" or "utility."
    """
    calculator: _RiskControlCalculator
    name: str = "coverage"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(default_factory=dict)

    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks: Callbacks) -> float:
        return self.calculator.get_scores()["coverage"]



def risk_control_suite(dataset: Dataset) -> list[Metric]:
    """
    Factory function to create the suite of four risk-control metrics.
    """
    calculator = _RiskControlCalculator(dataset)
    return [
        Risk(calculator=calculator),
        Carefulness(calculator=calculator),
        Alignment(calculator=calculator),
        Coverage(calculator=calculator),
    ]