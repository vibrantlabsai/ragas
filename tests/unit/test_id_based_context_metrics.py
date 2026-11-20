import math

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._context_precision import IDBasedContextPrecision
from ragas.metrics._context_recall import IDBasedContextRecall

# Hand-labeled retrieval fixture: for each query, the IDs a retriever returned
# and the IDs a human marked as relevant, with precision and recall worked out
# by hand.
#   id, retrieved, reference, expected_precision, expected_recall
KNOWN_ANSWERS = [
    # Documented examples (docs/concepts/metrics/available_metrics/).
    (
        "docs_precision",
        ["doc_1", "doc_2", "doc_3", "doc_4"],
        ["doc_1", "doc_4", "doc_5", "doc_6"],
        2 / 4,
        2 / 4,
    ),
    (
        "docs_recall",
        ["doc_1", "doc_2", "doc_3"],
        ["doc_1", "doc_4", "doc_5", "doc_6"],
        1 / 3,
        1 / 4,
    ),
    ("perfect", ["a", "b"], ["b", "a"], 1.0, 1.0),
    ("disjoint", ["x", "y"], ["a", "b"], 0.0, 0.0),
    ("superset_retrieved", ["a", "b", "c", "d", "e"], ["a", "c"], 2 / 5, 1.0),
    ("subset_retrieved", ["a"], ["a", "b", "c", "d"], 1.0, 1 / 4),
    ("int_ids", [101, 102, 103], [103, 104], 1 / 3, 1 / 2),
    ("int_matches_str", [1, 2, 3], ["1", "3"], 2 / 3, 1.0),
    ("duplicate_ids_counted_once", ["a", "a", "b"], ["a", "a"], 1 / 2, 1.0),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retrieved,reference,expected_precision,expected_recall",
    [case[1:] for case in KNOWN_ANSWERS],
    ids=[case[0] for case in KNOWN_ANSWERS],
)
async def test_id_based_context_metrics_known_answers(
    retrieved, reference, expected_precision, expected_recall
):
    sample = SingleTurnSample(
        retrieved_context_ids=retrieved, reference_context_ids=reference
    )

    precision = await IDBasedContextPrecision().single_turn_ascore(sample)
    recall = await IDBasedContextRecall().single_turn_ascore(sample)

    assert precision == pytest.approx(expected_precision)
    assert recall == pytest.approx(expected_recall)


@pytest.mark.asyncio
async def test_id_based_context_precision_is_nan_without_retrieved_ids():
    sample = SingleTurnSample(retrieved_context_ids=[], reference_context_ids=["a"])

    assert math.isnan(await IDBasedContextPrecision().single_turn_ascore(sample))
    assert await IDBasedContextRecall().single_turn_ascore(sample) == 0.0


@pytest.mark.asyncio
async def test_id_based_context_recall_is_nan_without_reference_ids():
    sample = SingleTurnSample(retrieved_context_ids=["a"], reference_context_ids=[])

    assert math.isnan(await IDBasedContextRecall().single_turn_ascore(sample))
    assert await IDBasedContextPrecision().single_turn_ascore(sample) == 0.0
