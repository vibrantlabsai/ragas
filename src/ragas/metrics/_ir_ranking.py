from __future__ import annotations

import logging
import math
import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricType, SingleTurnMetric
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


def _compute_jaccard_similarity(s1: str, s2: str) -> float:
    """Compute token-level Jaccard similarity between two context strings."""
    tokens1 = set(s1.lower().split())
    tokens2 = set(s2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return float(intersection / union) if union > 0 else 0.0


def _match_context(retrieved: str, references: t.List[str], threshold: float) -> bool:
    """Check if retrieved context matches any reference context."""
    retrieved_clean = retrieved.strip().lower()
    for ref in references:
        ref_clean = ref.strip().lower()
        if retrieved_clean == ref_clean or retrieved_clean in ref_clean or ref_clean in retrieved_clean:
            return True
        if threshold < 1.0 and _compute_jaccard_similarity(retrieved, ref) >= threshold:
            return True
    return False


@dataclass
class ContextNDCG(SingleTurnMetric):
    """
    Normalized Discounted Cumulative Gain at K (NDCG@K) for retrieved contexts.

    Measures whether ground-truth reference contexts appear at higher ranks in the
    retrieved context list. Evaluates the rank ordering quality of dense, sparse,
    or hybrid retrieval engines.

    Attributes
    ----------
    k : int
        Cutoff rank for evaluation (default: 10).
    similarity_threshold : float
        Jaccard threshold to consider a retrieved passage a match for a reference context (default: 0.7).
    """

    name: str = "context_ndcg"
    k: int = 10
    similarity_threshold: float = 0.7
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"retrieved_contexts", "reference_contexts"}
        }
    )

    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved = sample.retrieved_contexts or []
        references = sample.reference_contexts or []

        if not references:
            return 0.0
        if not retrieved:
            return 0.0

        cutoff = min(self.k, len(retrieved))
        dcg = 0.0
        for i in range(cutoff):
            rel = 1.0 if _match_context(retrieved[i], references, self.similarity_threshold) else 0.0
            if rel > 0.0:
                dcg += rel / math.log2(i + 2)

        # Ideal DCG: all true references ranked at the top
        ideal_matches = min(cutoff, len(references))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_matches))

        if idcg == 0.0:
            return 0.0

        return float(min(1.0, dcg / idcg))

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


@dataclass
class ContextMRR(SingleTurnMetric):
    """
    Mean Reciprocal Rank at K (MRR@K) for retrieved contexts.

    Computes 1 / rank of the first relevant retrieved context matching any reference context.
    Returns 0.0 if no reference context is retrieved within the top K.

    Attributes
    ----------
    k : int
        Cutoff rank for evaluation (default: 10).
    similarity_threshold : float
        Jaccard threshold to consider a retrieved passage a match (default: 0.7).
    """

    name: str = "context_mrr"
    k: int = 10
    similarity_threshold: float = 0.7
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"retrieved_contexts", "reference_contexts"}
        }
    )

    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved = sample.retrieved_contexts or []
        references = sample.reference_contexts or []

        if not references or not retrieved:
            return 0.0

        cutoff = min(self.k, len(retrieved))
        for i in range(cutoff):
            if _match_context(retrieved[i], references, self.similarity_threshold):
                return float(1.0 / (i + 1))

        return 0.0

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


@dataclass
class ContextMAP(SingleTurnMetric):
    """
    Mean Average Precision at K (MAP@K) for retrieved contexts.

    Computes the average precision across all recall cutoffs within the top K retrieved contexts.
    """

    name: str = "context_map"
    k: int = 10
    similarity_threshold: float = 0.7
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"retrieved_contexts", "reference_contexts"}
        }
    )

    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved = sample.retrieved_contexts or []
        references = sample.reference_contexts or []

        if not references or not retrieved:
            return 0.0

        cutoff = min(self.k, len(retrieved))
        hits = 0
        sum_precisions = 0.0

        for i in range(cutoff):
            if _match_context(retrieved[i], references, self.similarity_threshold):
                hits += 1
                sum_precisions += hits / (i + 1)

        denominator = min(cutoff, len(references))
        return float(sum_precisions / denominator) if denominator > 0 else 0.0

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


@dataclass
class ContextHitRate(SingleTurnMetric):
    """
    Hit Rate at K (HitRate@K) for retrieved contexts.

    Returns 1.0 if at least one reference context appears in the top K retrieved contexts,
    else 0.0.
    """

    name: str = "context_hit_rate"
    k: int = 10
    similarity_threshold: float = 0.7
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"retrieved_contexts", "reference_contexts"}
        }
    )

    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved = sample.retrieved_contexts or []
        references = sample.reference_contexts or []

        if not references or not retrieved:
            return 0.0

        cutoff = min(self.k, len(retrieved))
        for i in range(cutoff):
            if _match_context(retrieved[i], references, self.similarity_threshold):
                return 1.0

        return 0.0

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)
