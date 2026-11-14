"""Answer and retrieval consistency: agreement between repeated runs of the same input."""

import asyncio
import typing as t

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import (
    DIMENSION_WEIGHTS,
    AnswerAgreementInput,
    AnswerAgreementOutput,
    AnswerAgreementPrompt,
    jaccard,
    normalize_text,
    rescale,
    sample_run_pairs,
    weighted_agreement,
    word_ngrams,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


def _pairs(n: int) -> str:
    return "1 pair" if n == 1 else f"{n} pairs"


def _check_runs(runs: t.Any, name: str) -> t.List[t.Any]:
    if not isinstance(runs, (list, tuple)) or len(runs) < 2:
        raise ValueError(
            f"{name} must be a list of at least two runs of the same input"
        )
    return list(runs)


class AnswerConsistency(BaseMetric):
    """
    Measures how consistently a system answers the same input across repeated runs.

    No reference is needed: an LLM judge compares pairs of runs and rates how closely
    they agree on five dimensions (core facts, specifics such as numbers and names,
    supporting detail, stance, format), each from 1 to 5. The score is the weighted mean
    agreement, rescaled to 0.0-1.0 (1.0 = every compared pair says the same thing).

    Consistency is not correctness: a system that gives the same wrong answer every time
    scores 1.0. Use it alongside a correctness metric, or on its own where there are no
    labels.

    Comparing all N(N-1)/2 pairs of N runs is quadratic in LLM calls, so by default a random
    cycle of N pairs is judged, in which every run is compared at least twice. Identical
    runs (ignoring case and whitespace) are scored without an LLM call.

    Usage:
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import AnswerConsistency
        >>>
        >>> llm = llm_factory("gpt-4o-mini", client=AsyncOpenAI())
        >>> metric = AnswerConsistency(llm=llm)
        >>>
        >>> result = await metric.ascore(
        ...     user_input="How long are backups kept on the gold policy?",
        ...     responses=[
        ...         "Backups on the gold policy are kept for 30 days.",
        ...         "The gold policy retains backups for 30 days.",
        ...         "Gold policy backups are kept for 90 days.",
        ...     ],
        ... )
        >>> print(result.value, result.reason)

    Attributes:
        llm: Modern instructor-based LLM used as the judge
        name: The metric name
        max_comparisons: Number of run pairs to judge (default: number of runs)
        allowed_values: Score range (0.0 to 1.0, higher is more consistent)
    """

    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "answer_consistency",
        max_comparisons: t.Optional[int] = None,
        **kwargs,
    ):
        self.llm = llm
        self.max_comparisons = max_comparisons
        self.agreement_prompt = AnswerAgreementPrompt()
        super().__init__(name=name, **kwargs)

    async def ascore(
        self, responses: t.List[str], user_input: t.Optional[str] = None
    ) -> MetricResult:
        """
        Calculate answer consistency across repeated runs.

        Args:
            responses: The responses from two or more runs of the same input
            user_input: The input the runs answered (shown to the judge)

        Returns:
            MetricResult with the mean agreement (0.0-1.0) and a reason naming the least
            consistent dimension and the most different pair of runs
        """
        runs = [str(r) for r in _check_runs(responses, "responses")]
        pairs = sample_run_pairs(len(runs), self.max_comparisons)
        judged = await asyncio.gather(
            *(self._judge(user_input or "", runs[i], runs[j]) for i, j in pairs)
        )

        per_pair = [
            {d: rescale(getattr(v, d)) for d in DIMENSION_WEIGHTS} for v in judged
        ]
        pair_scores = [weighted_agreement(s) for s in per_pair]
        score = sum(pair_scores) / len(pair_scores)
        if score >= 1.0:
            return MetricResult(
                value=1.0,
                reason=f"All compared runs agree ({_pairs(len(pairs))} of {len(runs)} runs).",
            )

        dims = {
            d: sum(s[d] for s in per_pair) / len(per_pair) for d in DIMENSION_WEIGHTS
        }
        worst = min(range(len(pairs)), key=lambda k: pair_scores[k])
        i, j = pairs[worst]
        breakdown = ", ".join(f"{d} {v:.2f}" for d, v in dims.items())
        reason = (
            f"Mean agreement over {_pairs(len(pairs))} of {len(runs)} runs. "
            f"Per dimension: {breakdown}. Most different: runs {i + 1} and {j + 1} "
            f"({pair_scores[worst]:.2f}): {judged[worst].reason}"
        )
        return MetricResult(value=float(score), reason=reason)

    async def _judge(
        self, user_input: str, response_a: str, response_b: str
    ) -> AnswerAgreementOutput:
        if normalize_text(response_a) == normalize_text(response_b):
            return AnswerAgreementOutput(
                reason="The responses are identical.",
                **{d: 5 for d in DIMENSION_WEIGHTS},
            )
        prompt = self.agreement_prompt.to_string(
            AnswerAgreementInput(
                user_input=user_input, response_a=response_a, response_b=response_b
            )
        )
        return await self.llm.agenerate(prompt, AnswerAgreementOutput)


class RetrievalConsistency(BaseMetric):
    """
    Measures how consistently a retriever returns the same evidence for the same input.

    No relevance labels and no LLM are needed. Pass the results of two or more retrieval
    runs for one input, either as chunk IDs or as chunk texts:

    - ``retrieved_context_ids_runs``: runs are compared by the Jaccard similarity of their
      ID sets. Exact and cheap, but re-chunking or re-indexing changes the IDs.
    - ``retrieved_contexts_runs``: runs are compared by the Jaccard similarity of the word
      n-grams of their pooled text, so the same evidence split into different chunks still
      matches.

    Order within a run is ignored. The score is the mean similarity over compared pairs of
    runs (a random cycle of N pairs by default). Consistency is not relevance: a retriever
    that always returns the same irrelevant chunks scores 1.0.

    Usage:
        >>> from ragas.metrics.collections import RetrievalConsistency
        >>>
        >>> metric = RetrievalConsistency()
        >>> result = await metric.ascore(
        ...     retrieved_context_ids_runs=[
        ...         ["doc-12", "doc-7", "doc-3"],
        ...         ["doc-7", "doc-12", "doc-3"],
        ...         ["doc-12", "doc-7", "doc-41"],
        ...     ]
        ... )
        >>> print(result.value)

    Attributes:
        name: The metric name
        ngram_size: Word n-gram length used for chunk texts
        max_comparisons: Number of run pairs to compare (default: number of runs)
        allowed_values: Score range (0.0 to 1.0, higher is more consistent)
    """

    def __init__(
        self,
        name: str = "retrieval_consistency",
        ngram_size: int = 3,
        max_comparisons: t.Optional[int] = None,
        **kwargs,
    ):
        if ngram_size < 1:
            raise ValueError("ngram_size must be at least 1")
        self.ngram_size = ngram_size
        self.max_comparisons = max_comparisons
        super().__init__(name=name, **kwargs)

    async def ascore(
        self,
        retrieved_contexts_runs: t.Optional[t.List[t.List[str]]] = None,
        retrieved_context_ids_runs: t.Optional[
            t.List[t.List[t.Union[str, int]]]
        ] = None,
    ) -> MetricResult:
        """
        Calculate retrieval consistency across repeated runs.

        Args:
            retrieved_contexts_runs: For each run, the list of retrieved chunk texts
            retrieved_context_ids_runs: For each run, the list of retrieved chunk IDs

        Exactly one of the two must be given.

        Returns:
            MetricResult with the mean pairwise similarity (0.0-1.0)
        """
        if (retrieved_contexts_runs is None) == (retrieved_context_ids_runs is None):
            raise ValueError(
                "Provide exactly one of retrieved_contexts_runs or "
                "retrieved_context_ids_runs"
            )
        if retrieved_context_ids_runs is not None:
            runs = _check_runs(retrieved_context_ids_runs, "retrieved_context_ids_runs")
            sets: t.List[t.Set[t.Any]] = [set(run) for run in runs]
            unit = "chunk IDs"
        else:
            runs = _check_runs(retrieved_contexts_runs, "retrieved_contexts_runs")
            sets = [word_ngrams([str(c) for c in run], self.ngram_size) for run in runs]
            unit = f"word {self.ngram_size}-grams"

        pairs = sample_run_pairs(len(sets), self.max_comparisons)
        scores = [jaccard(sets[i], sets[j]) for i, j in pairs]
        score = sum(scores) / len(scores)
        worst = min(range(len(pairs)), key=lambda k: scores[k])
        i, j = pairs[worst]
        reason = (
            f"Mean Jaccard similarity of {unit} over {_pairs(len(pairs))} of "
            f"{len(runs)} runs; most different: runs {i + 1} and {j + 1} "
            f"({scores[worst]:.2f})."
        )
        return MetricResult(value=float(score), reason=reason)
