"""Regression tests for deterministic, non-mutating dataset splits."""

import random
from copy import deepcopy

import pytest
from pydantic import BaseModel

from ragas import Dataset, Experiment


class SplitRow(BaseModel):
    id: int


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = random.getstate()
    yield
    random.setstate(state)


def make_table(table_class, data_model):
    rows = [{"id": i} for i in range(10)]
    if data_model is not None:
        rows = [data_model(**row) for row in rows]
    return table_class("split_test", "inmemory", data_model=data_model, data=rows), rows


@pytest.mark.parametrize("table_class", [Dataset, Experiment])
@pytest.mark.parametrize("data_model", [None, SplitRow])
@pytest.mark.parametrize("seed", [0, 42])
def test_train_test_split_is_repeatable_and_preserves_input(
    table_class, data_model, seed
):
    table, rows = make_table(table_class, data_model)
    original = deepcopy(rows)

    train1, test1 = table.train_test_split(test_size=0.3, random_state=seed)
    train2, test2 = table.train_test_split(test_size=0.3, random_state=seed)

    assert list(train1) == list(train2)
    assert list(test1) == list(test2)
    assert list(table) == original
    assert rows == original
    assert len(train1) == 7
    assert len(test1) == 3
    train_ids = {table.get_row_value(row, "id") for row in train1}
    test_ids = {table.get_row_value(row, "id") for row in test1}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(range(10))


@pytest.mark.parametrize("seed", [None, 0, 42])
def test_train_test_split_preserves_global_random_state(seed):
    table, _ = make_table(Dataset, None)
    random.seed(2026)
    state = random.getstate()

    table.train_test_split(test_size=0.3, random_state=seed)

    assert random.getstate() == state


@pytest.mark.parametrize("table_class", [Dataset, Experiment])
@pytest.mark.parametrize("data_model", [None, SplitRow])
def test_train_test_split_without_seed_keeps_input_order(table_class, data_model):
    table, rows = make_table(table_class, data_model)
    original = deepcopy(rows)

    train, test = table.train_test_split(test_size=0.3)

    assert list(train) == original[:7]
    assert list(test) == original[7:]
    assert list(table) == original
