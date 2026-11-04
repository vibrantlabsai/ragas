"""Before/after RAG evaluation through the ragas ``Dataset`` + ``@experiment`` API."""

from __future__ import annotations

import asyncio
import typing as t
from pathlib import Path

from ragas.optimizers.prompt_optimization.budget import BudgetExceeded
from ragas.optimizers.prompt_optimization.bundle import GoldenRow, PromptBundle
from ragas.optimizers.prompt_optimization.metrics import Scorer, aggregate
from ragas.optimizers.prompt_optimization.rag_evaluation import (
    AnswerFn,
    EvaluationReport,
    RowResult,
)


async def _arun_ragas_eval(
    bundle: PromptBundle,
    rows: t.Sequence[GoldenRow],
    answer_fn: AnswerFn,
    scorer: Scorer,
    root_dir: Path,
    name: str,
    split: str,
) -> t.Tuple[EvaluationReport, Path]:
    from ragas import Dataset, experiment

    by_id = {r.id: r for r in rows}
    dataset = Dataset(
        name=f"{name}_dataset", backend="local/csv", root_dir=str(root_dir)
    )
    for row in rows:
        dataset.append(
            {
                "id": row.id,
                "question": row.question,
                "context": row.context_text,
                "reference": row.reference,
            }
        )
    dataset.save()

    def answer_and_score(golden: GoldenRow) -> t.Tuple[str, str, t.Dict[str, float]]:
        # Runs in a worker thread: ragas metrics' sync score() refuses to run
        # inside an event loop, and the student call is blocking I/O.
        try:
            answer, error = answer_fn(bundle, golden), ""
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001 - recorded, scored as a failure
            answer, error = "", f"{type(e).__name__}: {e}"
        return answer, error, scorer.score(golden, answer)

    @experiment()
    async def rag_experiment(sample: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        golden = by_id[str(sample["id"])]
        answer, error, scores = await asyncio.to_thread(answer_and_score, golden)
        return {
            **sample,
            "prompt_source": bundle.source,
            "prompt_fingerprint": bundle.fingerprint(),
            "response": answer,
            "error": error,
            **scores,
        }

    results = await rag_experiment.arun(dataset, name=name)
    results.save()

    result_rows = {str(r["id"]): r for r in results}
    missing = [row.id for row in rows if row.id not in result_rows]
    if missing:
        raise RuntimeError(
            f"ragas experiment {name!r} returned no result for {len(missing)} row(s): "
            f"{missing[:5]}. See the warnings above for the task errors."
        )
    ordered = [
        RowResult(
            row_id=row.id,
            answer=str(result_rows[row.id]["response"]),
            scores={m: float(result_rows[row.id][m]) for m in scorer.metric_names},
            error=str(result_rows[row.id].get("error") or "") or None,
        )
        for row in rows
    ]
    report = EvaluationReport(
        source=bundle.source,
        fingerprint=bundle.fingerprint(),
        split=split,
        rows=ordered,
        aggregates=aggregate([r.scores for r in ordered]),
    )
    return report, root_dir / "experiments" / f"{name}.csv"


def run_ragas_eval(
    bundle: PromptBundle,
    rows: t.Sequence[GoldenRow],
    answer_fn: AnswerFn,
    scorer: Scorer,
    root_dir: t.Union[str, Path],
    name: str,
    split: str = "holdout",
) -> t.Tuple[EvaluationReport, Path]:
    """Run the RAG app with ``bundle`` on ``rows`` as a ragas experiment.

    Returns the scores and the path of the saved experiment CSV.
    """
    return asyncio.run(
        _arun_ragas_eval(bundle, rows, answer_fn, scorer, Path(root_dir), name, split)
    )
