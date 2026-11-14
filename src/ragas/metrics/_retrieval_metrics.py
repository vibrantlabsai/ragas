"""Traditional information-retrieval metrics: NDCG, MRR, P@K, R@K.

These metrics evaluate retrieval quality without an LLM judge.  They compare
the ranked list of retrieved document IDs against a set of relevant document
IDs and return a scalar score in [0, 1].

Usage::

    from ragas.metrics import NDCGScore, MRRScore, PrecisionAtK, RecallAtK
    from ragas.dataset_schema import SingleTurnSample

    sample = SingleTurnSample(
        user_input="What causes transformer attention to scale quadratically?",
        retrieved_context_ids=["doc_1", "doc_3", "doc_7", "doc_2", "doc_9"],
        reference_context_ids=["doc_1", "doc_7"],
    )

    ndcg = NDCGScore(k=5)
    print(ndcg.single_turn_score(sample))   # 0.9307

    mrr = MRRScore()
    print(mrr.single_turn_score(sample))    # 1.0

References:
    Järvelin & Kekäläinen, "Cumulated gain-based evaluation of IR techniques",
    ACM TOIS, 2002.  https://dl.acm.org/doi/10.1145/582415.582418
"""
from __future__ import annotations

import math
import typing as t
from dataclasses import dataclass, field

from langchain_core.callbacks import Callbacks

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricType, SingleTurnMetric
from ragas.run_config import RunConfig


def _dcg(relevances: list[float], k: int | None = None) -> float:
    """Discounted Cumulative Gain."""
    rels = relevances[:k] if k is not None else relevances
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(rels))


def _ndcg(retrieved: list, relevant: set, k: int | None = None) -> float:
    """NDCG@K — normalized against the ideal DCG."""
    rels = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved]
    ideal = sorted(rels, reverse=True)
    idcg = _dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return _dcg(rels, k) / idcg


def _reciprocal_rank(retrieved: list, relevant: set) -> float:
    """Reciprocal rank of the first relevant document."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Fraction of the top-K retrieved documents that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for doc_id in top_k if doc_id in relevant) / len(top_k)


def _recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Fraction of relevant documents captured in the top-K results."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for doc_id in top_k if doc_id in relevant) / len(relevant)


def _validate_sample(sample: SingleTurnSample) -> tuple[list, set]:
    """Extract and validate retrieved / reference context ID lists."""
    if sample.retrieved_context_ids is None:
        raise ValueError(
            "SingleTurnSample.retrieved_context_ids is required for IR metrics. "
            "Pass an ordered list of retrieved document IDs."
        )
    if sample.reference_context_ids is None:
        raise ValueError(
            "SingleTurnSample.reference_context_ids is required for IR metrics. "
            "Pass the set of relevant document IDs."
        )
    return list(sample.retrieved_context_ids), set(sample.reference_context_ids)


@dataclass
class NDCGScore(SingleTurnMetric):
    """Normalized Discounted Cumulative Gain (NDCG@K).

    Measures ranking quality: a relevant document at rank 1 contributes more
    than the same document at rank 5.  Score is 1.0 when the ideal ranking
    is returned, 0.0 when no relevant document is retrieved.

    Parameters
    ----------
    k : int, optional
        Cutoff rank.  Only the top-k retrieved documents are considered.
        ``None`` evaluates over the full ranked list.

    Required sample fields
    ----------------------
    retrieved_context_ids : List[str | int]
        Ordered list of retrieved document IDs (best first).
    reference_context_ids : List[str | int]
        List (or set) of relevant document IDs.
    """

    name: str = "ndcg_score"
    k: t.Optional[int] = None
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"retrieved_context_ids", "reference_context_ids"}
        }
    )

    def init(self, run_config: RunConfig) -> None:
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved, relevant = _validate_sample(sample)
        return _ndcg(retrieved, relevant, k=self.k)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


@dataclass
class MRRScore(SingleTurnMetric):
    """Mean Reciprocal Rank (MRR).

    Returns 1/rank of the first relevant document in the ranked list.
    0.0 if no relevant document is retrieved.

    Required sample fields
    ----------------------
    retrieved_context_ids : List[str | int]
    reference_context_ids : List[str | int]
    """

    name: str = "mrr_score"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"retrieved_context_ids", "reference_context_ids"}
        }
    )

    def init(self, run_config: RunConfig) -> None:
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved, relevant = _validate_sample(sample)
        return _reciprocal_rank(retrieved, relevant)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


@dataclass
class PrecisionAtK(SingleTurnMetric):
    """Precision@K — fraction of top-K retrieved documents that are relevant.

    Parameters
    ----------
    k : int
        Rank cutoff (required).

    Required sample fields
    ----------------------
    retrieved_context_ids : List[str | int]
    reference_context_ids : List[str | int]
    """

    name: str = "precision_at_k"
    k: int = 5
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"retrieved_context_ids", "reference_context_ids"}
        }
    )

    def init(self, run_config: RunConfig) -> None:
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved, relevant = _validate_sample(sample)
        return _precision_at_k(retrieved, relevant, k=self.k)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


@dataclass
class RecallAtK(SingleTurnMetric):
    """Recall@K — fraction of relevant documents captured in top-K results.

    Parameters
    ----------
    k : int
        Rank cutoff (required).

    Required sample fields
    ----------------------
    retrieved_context_ids : List[str | int]
    reference_context_ids : List[str | int]
    """

    name: str = "recall_at_k"
    k: int = 5
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"retrieved_context_ids", "reference_context_ids"}
        }
    )

    def init(self, run_config: RunConfig) -> None:
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved, relevant = _validate_sample(sample)
        return _recall_at_k(retrieved, relevant, k=self.k)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)
