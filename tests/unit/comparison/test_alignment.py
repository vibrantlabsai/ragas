import pytest

from ragas.comparison import ComparisonPlan, ExpectedCase, MetricSpec, SliceSpec
from ragas.comparison.alignment import align
from ragas.comparison.observations import capture


def aligned(before, after, **kwargs):
    plan = ComparisonPlan(metrics=(MetricSpec("score"),), **kwargs)
    return align(capture(before, plan), capture(after, plan), plan)


def test_reordering_pairs_ids_and_keeps_case_weights_equal():
    before = [{"case_id": "a", "score": 0}, {"case_id": "b", "score": 1}]
    after = [{"case_id": "b", "score": 0}, {"case_id": "a", "score": 1}]
    cohort = aligned(before, after)
    assert [(p.case_id, p.baseline, p.candidate) for p in cohort.pairs[0]] == [
        ("a", (0.0,), (1.0,)),
        ("b", (1.0,), (0.0,)),
    ]
    assert cohort.provenance == "observed_union"


def test_manifest_covers_absent_both_and_freezes_slices():
    manifest = (
        ExpectedCase("a", attributes=(("lang", "en"),)),
        ExpectedCase("b", attributes=(("lang", "en"),)),
    )
    row = {"case_id": "a", "score": 1, "lang": "en"}
    cohort = aligned(
        [row],
        [row],
        expected_cases=manifest,
        slices=(SliceSpec("english", "lang", ("en",)),),
    )
    assert cohort.provenance == "expected_manifest"
    assert cohort.groups[1].case_ids == ("a", "b")
    assert cohort.pairs[0][1].reasons == ("baseline_absent", "candidate_absent")


def test_failed_candidate_remains_in_slice_denominator():
    before = [{"case_id": "a", "score": 0, "lang": "en", "state": "success"}]
    after = [{"case_id": "a", "score": 1, "lang": "en", "state": "error"}]
    cohort = aligned(
        before,
        after,
        status_field="state",
        slices=(SliceSpec("english", "lang", ("en",)),),
    )
    assert cohort.groups[1].case_ids == ("a",)
    assert not cohort.pairs[0][0].eligible
    assert "candidate_unsuccessful" in cohort.pairs[0][0].reasons


def test_repeat_mismatch_excludes_whole_case():
    before = [
        {"case_id": "a", "r": "1", "score": 0},
        {"case_id": "a", "r": "2", "score": 0},
    ]
    after = [{"case_id": "a", "r": "1", "score": 1}]
    pair = aligned(before, after, repeat_id="r").pairs[0][0]
    assert not pair.eligible
    assert "candidate_repeat_absent" in pair.reasons
    assert pair.expected_repeats == 2


def test_missing_metric_is_metric_specific():
    plan = ComparisonPlan(metrics=(MetricSpec("a"), MetricSpec("b")))
    before = [{"case_id": "one", "a": 1, "b": 1}]
    after = [{"case_id": "one", "a": 1, "b": None}]
    cohort = align(capture(before, plan), capture(after, plan), plan)
    assert cohort.pairs[0][0].eligible
    assert not cohort.pairs[1][0].eligible


@pytest.mark.parametrize("baseline,candidate", [(True, 1), (1, 1.0), ("x", "y")])
def test_changed_identity_is_not_a_pair(baseline, candidate):
    with pytest.raises(ValueError, match="attribute"):
        aligned(
            [{"case_id": "a", "score": 1, "input": baseline}],
            [{"case_id": "a", "score": 1, "input": candidate}],
            case_fields=("input",),
        )


def test_repeats_must_agree_on_slice_attributes_even_when_failed():
    rows = [
        {"case_id": "a", "r": "1", "score": None, "lang": "en"},
        {"case_id": "a", "r": "2", "score": None, "lang": "de"},
    ]
    with pytest.raises(ValueError, match="attribute"):
        aligned(rows, [], repeat_id="r", slices=(SliceSpec("en", "lang", ("en",)),))


def test_manifest_rejects_unexpected_cases_and_repeats():
    with pytest.raises(ValueError, match="unexpected"):
        aligned([{"case_id": "x", "score": 1}], [], expected_cases=(ExpectedCase("a"),))
    with pytest.raises(ValueError, match="unexpected"):
        aligned(
            [{"case_id": "a", "r": "2", "score": 1}],
            [],
            repeat_id="r",
            expected_cases=(ExpectedCase("a", repeats=("1",)),),
        )


def test_empty_groups_stay_in_the_fixed_family():
    row = {"case_id": "a", "score": 1, "lang": "en"}
    cohort = aligned([row], [row], slices=(SliceSpec("de", "lang", ("de",)),))
    assert cohort.groups[1].case_ids == ()


def test_alignment_allocation_is_bounded_before_building_pairs():
    from ragas.comparison import ComparisonLimits

    plan = ComparisonPlan(metrics=(MetricSpec("a"), MetricSpec("b")))
    rows = [{"case_id": "one", "a": 1, "b": 1}, {"case_id": "two", "a": 1, "b": 1}]
    snapshot = capture(rows, plan)
    with pytest.raises(ValueError, match="alignment"):
        align(snapshot, snapshot, plan, ComparisonLimits(max_alignment_cells=2))
