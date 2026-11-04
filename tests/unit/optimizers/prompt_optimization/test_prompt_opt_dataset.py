"""R4: golden rows load, map ragas field names, and split into four disjoint sets."""

import json

import pytest
from prompt_opt_testdata import GOLDEN_DATASET

from ragas.optimizers.prompt_optimization.bundle import SPLITS
from ragas.optimizers.prompt_optimization.dataset import (
    DatasetError,
    load_golden_rows,
    split_rows,
    to_dspy_example,
)


def _row(**overrides):
    row = {
        "id": "r1",
        "user_input": "Q?",
        "retrieved_contexts": ["ctx"],
        "reference": "A",
        "split": "train",
    }
    row.update(overrides)
    return row


def test_loads_shipped_golden_dataset():
    rows = load_golden_rows(GOLDEN_DATASET)
    assert len(rows) == 80
    splits = split_rows(rows)
    assert {name: len(v) for name, v in splits.items()} == {s: 20 for s in SPLITS}
    for name in SPLITS:
        assert sum(not r.answerable for r in splits[name]) == 3


def test_splits_are_disjoint(golden_rows):
    splits = split_rows(golden_rows)
    ids = [r.id for name in SPLITS for r in splits[name]]
    assert len(ids) == len(set(ids)) == len(golden_rows)


def test_field_aliases_and_jsonl(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                _row(),
                {
                    "id": "r2",
                    "question": "Q2",
                    "contexts": "single ctx",
                    "answer": "A2",
                    "split": "eval",
                },
            ]
        ),
        encoding="utf-8",
    )
    rows = load_golden_rows(path)
    assert rows[1].question == "Q2"
    assert rows[1].contexts == ("single ctx",)
    assert rows[1].reference == "A2"


def test_loads_json_list_and_dicts(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([_row()]), encoding="utf-8")
    assert load_golden_rows(path)[0].id == "r1"
    assert load_golden_rows([_row(id="x")])[0].id == "x"


def test_json_must_be_list(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"rows": []}), encoding="utf-8")
    with pytest.raises(DatasetError):
        load_golden_rows(path)


@pytest.mark.parametrize(
    "missing", ["user_input", "retrieved_contexts", "reference", "split"]
)
def test_missing_fields_rejected(missing):
    row = _row()
    del row[missing]
    with pytest.raises(DatasetError, match=missing):
        load_golden_rows([row])


def test_bad_split_and_duplicate_ids_rejected():
    with pytest.raises(DatasetError, match="split"):
        load_golden_rows([_row(split="validation")])
    with pytest.raises(DatasetError, match="duplicate"):
        load_golden_rows([_row(), _row()])


def test_empty_required_split_rejected(golden_rows):
    only_train = [r for r in golden_rows if r.split == "train"]
    with pytest.raises(DatasetError, match="eval"):
        split_rows(only_train)
    assert split_rows(only_train, required=["train"])["train"] == only_train


def test_answerable_false_is_kept():
    assert load_golden_rows([_row(answerable=False)])[0].answerable is False


def test_to_dspy_example(golden_rows):
    pytest.importorskip("dspy")
    example = to_dspy_example(golden_rows[0])
    assert example.row_id == golden_rows[0].id
    assert set(example.inputs().keys()) == {"context", "question"}
    assert example.answer == golden_rows[0].reference
    assert example.context.startswith("[1] ")
