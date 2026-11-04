"""TokenOverlap metric - Modern collections implementation."""

import re
import string
import typing as t
from collections import Counter

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

TokenOverlapMode = t.Literal["precision", "recall", "f1", "fbeta"]

_PUNCTUATION = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    """Normalize text the way SQuAD-style QA evaluation does.

    Lowercases, strips punctuation, removes the articles "a", "an" and "the",
    and collapses whitespace.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def f_beta(precision: float, recall: float, beta: float) -> float:
    """F-beta score. ``beta < 1`` weights precision more, ``beta > 1`` recall more."""
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def token_precision_recall(reference: str, response: str) -> t.Tuple[float, float]:
    """Token-level precision and recall after SQuAD normalization.

    Both empty counts as a perfect match; exactly one empty scores zero.
    """
    ref_tokens = normalize_answer(reference).split()
    resp_tokens = normalize_answer(response).split()
    if not ref_tokens and not resp_tokens:
        return 1.0, 1.0
    if not ref_tokens or not resp_tokens:
        return 0.0, 0.0
    overlap = sum((Counter(ref_tokens) & Counter(resp_tokens)).values())
    return overlap / len(resp_tokens), overlap / len(ref_tokens)


class TokenOverlap(BaseMetric):
    """
    Token-level overlap between a response and a reference answer.

    Text is normalized SQuAD-style (lowercase, no punctuation, no articles) and
    split on whitespace. Overlap counts shared tokens, respecting multiplicity.

    Modes:
        - ``precision``: share of response tokens that appear in the reference
        - ``recall``: share of reference tokens that appear in the response
        - ``f1``: harmonic mean of precision and recall
        - ``fbeta``: F-beta with the given ``beta`` (``beta=0.5`` favours precision)

    This metric does not call an LLM or an embedding model.

    Usage:
        >>> from ragas.metrics.collections import TokenOverlap
        >>>
        >>> metric = TokenOverlap(mode="fbeta", beta=0.5)
        >>> result = await metric.ascore(
        ...     reference="15 days per year",
        ...     response="New employees get 15 days of PTO per year.",
        ... )
        >>> print(f"F0.5: {result.value:.3f}")

    Attributes:
        name: The metric name (defaults to ``token_<mode>``, or ``token_f<beta>``)
        mode: Which overlap statistic to return
        beta: The beta used when ``mode="fbeta"``
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(
        self,
        mode: TokenOverlapMode = "f1",
        beta: float = 1.0,
        name: t.Optional[str] = None,
        **base_kwargs,
    ):
        """Initialize TokenOverlap metric."""
        if mode not in ("precision", "recall", "f1", "fbeta"):
            raise ValueError(
                f"mode must be one of 'precision', 'recall', 'f1', 'fbeta'; got {mode!r}"
            )
        if beta <= 0:
            raise ValueError(f"beta must be positive; got {beta}")
        if name is None:
            name = (
                f"token_f{beta:g}".replace(".", "_")
                if mode == "fbeta"
                else f"token_{mode}"
            )
        super().__init__(name=name, **base_kwargs)
        self.mode = mode
        self.beta = beta

    async def ascore(
        self,
        reference: str,
        response: str,
    ) -> MetricResult:
        """
        Calculate the token overlap score asynchronously.

        Args:
            reference: The reference/ground truth text
            response: The response text to evaluate

        Returns:
            MetricResult with the score (0.0-1.0)
        """
        if not isinstance(reference, str) or not isinstance(response, str):
            return MetricResult(
                value=0.0,
                reason="Invalid input: reference and response must be strings",
            )

        precision, recall = token_precision_recall(reference, response)
        if self.mode == "precision":
            score = precision
        elif self.mode == "recall":
            score = recall
        elif self.mode == "f1":
            score = f_beta(precision, recall, 1.0)
        else:
            score = f_beta(precision, recall, self.beta)

        return MetricResult(value=float(score))
