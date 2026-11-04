"""Load and split a validated golden RAG dataset."""

from __future__ import annotations

import json
import typing as t
from pathlib import Path

from ragas.optimizers.prompt_optimization.bundle import SPLITS, GoldenRow

RowSource = t.Union[str, Path, t.Sequence[t.Mapping[str, t.Any]], t.Any]


class DatasetError(ValueError):
    """Raised when a golden dataset is missing fields or splits."""


def _rows_from_source(source: RowSource) -> t.List[t.Mapping[str, t.Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        data = json.loads(text)
        if not isinstance(data, list):
            raise DatasetError(f"{path} must contain a JSON list of rows")
        return data
    # ragas Dataset / EvaluationDataset and similar iterables of samples
    any_source: t.Any = source
    if hasattr(any_source, "to_pandas") and not isinstance(any_source, list):
        return any_source.to_pandas().to_dict(orient="records")
    rows: t.List[t.Mapping[str, t.Any]] = []
    for item in any_source:
        model_dump = getattr(item, "model_dump", None)
        row = model_dump() if callable(model_dump) else item
        rows.append(t.cast(t.Mapping[str, t.Any], row))
    return rows


def _field(row: t.Mapping[str, t.Any], *names: str) -> t.Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def load_golden_rows(source: RowSource) -> t.List[GoldenRow]:
    """Read golden rows from a JSONL/JSON file, a ragas dataset, or a list of dicts.

    Accepted field names follow ragas ``SingleTurnSample`` (``user_input``,
    ``retrieved_contexts``, ``reference``) with ``question``/``contexts``/
    ``answer`` as aliases. Every row needs a ``split`` of train, eval, test or
    holdout.
    """
    rows = _rows_from_source(source)
    golden: t.List[GoldenRow] = []
    seen_ids: t.Set[str] = set()
    for index, row in enumerate(rows):
        row_id = str(_field(row, "id", "row_id") or f"row{index:03d}")
        question = _field(row, "user_input", "question")
        contexts = _field(row, "retrieved_contexts", "contexts")
        reference = _field(row, "reference", "answer", "ground_truth")
        split = _field(row, "split")
        missing = [
            name
            for name, value in (
                ("user_input", question),
                ("retrieved_contexts", contexts),
                ("reference", reference),
                ("split", split),
            )
            if value is None
        ]
        if missing:
            raise DatasetError(f"row {row_id!r} is missing {', '.join(missing)}")
        if split not in SPLITS:
            raise DatasetError(
                f"row {row_id!r} has split {split!r}; expected one of {', '.join(SPLITS)}"
            )
        if isinstance(contexts, str):
            contexts = [contexts]
        if row_id in seen_ids:
            raise DatasetError(f"duplicate row id {row_id!r}")
        seen_ids.add(row_id)
        golden.append(
            GoldenRow(
                id=row_id,
                question=str(question),
                contexts=tuple(str(c) for c in contexts),
                reference=str(reference),
                split=str(split),
                answerable=bool(
                    _field(row, "answerable") if "answerable" in row else True
                ),
            )
        )
    return golden


def split_rows(
    rows: t.Sequence[GoldenRow],
    required: t.Sequence[str] = SPLITS,
) -> t.Dict[str, t.List[GoldenRow]]:
    """Group rows by split and check that each required split is non-empty."""
    splits: t.Dict[str, t.List[GoldenRow]] = {name: [] for name in SPLITS}
    for row in rows:
        splits[row.split].append(row)
    empty = [name for name in required if not splits[name]]
    if empty:
        raise DatasetError(f"dataset has no rows in split(s): {', '.join(empty)}")
    return splits


def to_dspy_example(row: GoldenRow) -> t.Any:
    """Convert a golden row to a ``dspy.Example`` with context/question as inputs."""
    import dspy

    return dspy.Example(
        row_id=row.id,
        context=row.context_text,
        question=row.question,
        answer=row.reference,
    ).with_inputs("context", "question")
