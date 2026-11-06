import math

import pytest
from pydantic import BaseModel

from ragas.comparison import (
    ComparisonLimits,
    ComparisonPlan,
    ExpectedCase,
    MetricSpec,
    SliceSpec,
)
from ragas.comparison.observations import capture


def plan(**kwargs):
    return ComparisonPlan(metrics=(MetricSpec("quality"),), **kwargs)


def test_capture_is_order_independent_and_does_not_retain_unselected_payload():
    rows = [
        {"case_id": "b", "quality": 0.7, "secret": object()},
        {"case_id": "a", "quality": 0.4},
    ]
    first = capture(rows, plan())
    second = capture(reversed(rows), plan())
    assert first == second
    assert first.rows[0].case_id == "a"
    assert "secret" not in repr(first)


def test_typed_models_and_mappings_have_equal_snapshots():
    class Row(BaseModel):
        case_id: str
        quality: float

    assert capture([Row(case_id="a", quality=1)], plan()) == capture(
        [{"case_id": "a", "quality": 1.0}], plan()
    )


@pytest.mark.parametrize("value", [True, "0.5", float("inf"), 10**400, []])
def test_invalid_scores_reject_instead_of_becoming_missing(value):
    with pytest.raises(ValueError):
        capture([{"case_id": "a", "quality": value}], plan())


@pytest.mark.parametrize(
    "row,reason",
    [
        ({"case_id": "a"}, "missing"),
        ({"case_id": "a", "quality": None}, "null"),
        ({"case_id": "a", "quality": math.nan}, "nan"),
    ],
)
def test_missing_scores_retain_exclusion_reason(row, reason):
    captured = capture([row], plan())
    assert captured.rows[0].scores[0].value is None
    assert captured.rows[0].scores[0].reason == reason


def test_failed_status_does_not_allow_a_present_score_to_count():
    captured = capture(
        [{"case_id": "a", "quality": 1, "state": "error"}], plan(status_field="state")
    )
    assert captured.rows[0].successful is False


@pytest.mark.parametrize("status", [None, 1, True, "unknown"])
def test_unrecognized_status_is_not_success(status):
    assert (
        not capture(
            [{"case_id": "a", "quality": 1, "state": status}],
            plan(status_field="state"),
        )
        .rows[0]
        .successful
    )


def test_duplicate_keys_and_missing_selected_repeat_reject():
    row = {"case_id": "a", "quality": 1}
    with pytest.raises(ValueError, match="duplicate"):
        capture([row, row], plan())
    with pytest.raises(ValueError, match="repeat"):
        capture([row], plan(repeat_id="repeat"))


def test_missing_selected_attributes_and_type_coercion_reject():
    p = plan(
        case_fields=("input",), slices=(SliceSpec("english", "language", ("en",)),)
    )
    with pytest.raises(ValueError, match="field"):
        capture([{"case_id": "a", "quality": 1, "language": "en"}], p)
    with pytest.raises(ValueError, match="case_id"):
        capture([{"case_id": 1, "quality": 1}], plan())


def test_limits_count_iterables_and_selected_strings():
    with pytest.raises(ValueError, match="rows"):
        capture(
            ({"case_id": str(i), "quality": 1} for i in range(10)),
            plan(),
            ComparisonLimits(max_rows=2),
        )
    with pytest.raises(ValueError, match="string"):
        capture(
            [{"case_id": "long", "quality": 1}],
            plan(),
            ComparisonLimits(max_string_bytes=2),
        )


def test_range_and_nonvacuous_configuration():
    with pytest.raises(ValueError):
        ComparisonPlan(metrics=())
    with pytest.raises(ValueError):
        ComparisonPlan(metrics=(MetricSpec("x", required=False),))
    with pytest.raises(ValueError):
        plan(expected_cases=())
    with pytest.raises(ValueError):
        MetricSpec("x", margin=True)
    with pytest.raises(ValueError):
        capture(
            [{"case_id": "a", "quality": 1.1}],
            ComparisonPlan(metrics=(MetricSpec("quality", minimum=0, maximum=1),)),
        )


def test_manifest_preserves_typed_attributes_and_repeats():
    case = ExpectedCase("a", repeats=("r1", "r2"), attributes=(("tenant", True),))
    p = plan(repeat_id="repeat", case_fields=("tenant",), expected_cases=(case,))
    assert p.expected_cases[0].attributes == (("tenant", True),)


def test_unrepresentable_integer_score_is_not_rounded_into_a_tie():
    with pytest.raises(ValueError, match="exact"):
        capture([{"case_id": "a", "quality": 2**53 + 1}], plan())


def test_explicit_empty_metric_column_is_invalid():
    with pytest.raises(ValueError):
        MetricSpec("quality", column="")


def test_manifest_resource_limits_apply_even_to_absent_cases():
    p = plan(expected_cases=(ExpectedCase("too-long"),))
    with pytest.raises(ValueError, match="string"):
        capture([], p, ComparisonLimits(max_string_bytes=4))
    repeated = plan(
        repeat_id="r",
        expected_cases=(ExpectedCase("a", ("1", "2")), ExpectedCase("b", ("1", "2"))),
    )
    with pytest.raises(ValueError, match="rows"):
        capture([], repeated, ComparisonLimits(max_rows=3))
