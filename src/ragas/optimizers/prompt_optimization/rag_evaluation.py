"""RAG evaluation of a prompt bundle: run the student on a split and score it."""

from __future__ import annotations

import logging
import typing as t
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ragas.optimizers.prompt_optimization.budget import BudgetExceeded
from ragas.optimizers.prompt_optimization.bundle import GoldenRow, PromptBundle
from ragas.optimizers.prompt_optimization.metrics import Scorer, aggregate

logger = logging.getLogger(__name__)

#: Produces the student's answer for one row under a given prompt bundle.
AnswerFn = t.Callable[[PromptBundle, GoldenRow], str]


@dataclass
class RowResult:
    row_id: str
    answer: str
    scores: t.Dict[str, float]
    error: t.Optional[str] = None


@dataclass
class EvaluationReport:
    """Per-row and aggregate scores of one bundle on one split."""

    source: str
    fingerprint: str
    split: str
    rows: t.List[RowResult] = field(default_factory=list)
    aggregates: t.Dict[str, float] = field(default_factory=dict)

    def scores_by_row(self, metric: str) -> t.Dict[str, float]:
        return {r.row_id: r.scores[metric] for r in self.rows}

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            "source": self.source,
            "fingerprint": self.fingerprint,
            "split": self.split,
            "aggregates": self.aggregates,
            "rows": [
                {
                    "row_id": r.row_id,
                    "answer": r.answer,
                    "scores": r.scores,
                    "error": r.error,
                }
                for r in self.rows
            ],
        }


def evaluate_bundle(
    bundle: PromptBundle,
    rows: t.Sequence[GoldenRow],
    answer_fn: AnswerFn,
    scorer: Scorer,
    split: str = "",
    num_threads: int = 8,
) -> EvaluationReport:
    """Answer every row with ``bundle`` and score the answers.

    A failed call (other than a budget stop) is recorded with an empty answer,
    which scores as a failure instead of crashing the whole evaluation.
    """

    def run(row: GoldenRow) -> RowResult:
        try:
            answer = answer_fn(bundle, row)
            error = None
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001 - one bad row must not stop the run
            logger.warning("row %s failed: %s", row.id, e)
            answer, error = "", f"{type(e).__name__}: {e}"
        if not isinstance(answer, str):
            answer, error = "", "answer was not a string"
        return RowResult(row.id, answer, scorer.score(row, answer), error)

    if num_threads <= 1 or len(rows) <= 1:
        results = [run(row) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            results = list(pool.map(run, rows))

    return EvaluationReport(
        source=bundle.source,
        fingerprint=bundle.fingerprint(),
        split=split,
        rows=results,
        aggregates=aggregate([r.scores for r in results]),
    )
