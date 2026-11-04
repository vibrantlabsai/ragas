"""TokenOverlap and ExactMatch(normalize=True) in ragas.metrics.collections."""

import statistics

import pytest

from ragas.metrics.collections import ExactMatch, TokenOverlap
from ragas.metrics.collections.token_overlap import (
    f_beta,
    normalize_answer,
    token_precision_recall,
)

# The four val rows of the prototype dataset (reference, response).
PROTOTYPE_VAL_ROWS = [
    (
        "It will not be reimbursed",
        "Late reports may still be reimbursed with your manager's approval, but it is "
        "best to submit within 30 days to avoid delays.",
    ),
    ("Annually, within 30 days of the hire anniversary", "Annually."),
    (
        "I don't know based on the provided documents.",
        "I don't know based on the provided documents.",
    ),
    (
        "8 sick days per year",
        "You get 8 sick days per year. If you need more you can use your PTO, and some "
        "states also require additional paid sick leave by law.",
    ),
]


def test_normalize_answer_squad_style():
    assert normalize_answer("  The Answer, is: A test! ") == "answer is test"


@pytest.mark.parametrize(
    "precision,recall,beta,expected",
    [
        (0.5, 0.5, 1.0, 0.5),
        (1.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.5, 0.0),
        (1.0, 0.5, 0.5, 0.833333),
    ],
)
def test_f_beta(precision, recall, beta, expected):
    assert f_beta(precision, recall, beta) == pytest.approx(expected, abs=1e-6)


def test_precision_recall_counts_multiplicity():
    p, r = token_precision_recall("days days off", "days off")
    assert (p, r) == (pytest.approx(1.0), pytest.approx(2 / 3))


def test_precision_recall_empty_cases():
    assert token_precision_recall("", "") == (1.0, 1.0)
    assert token_precision_recall("a b", "") == (0.0, 0.0)


@pytest.mark.parametrize(
    "mode,beta,expected_name",
    [
        ("precision", 1.0, "token_precision"),
        ("recall", 1.0, "token_recall"),
        ("f1", 1.0, "token_f1"),
        ("fbeta", 0.5, "token_f0_5"),
        ("fbeta", 2.0, "token_f2"),
    ],
)
def test_default_names(mode, beta, expected_name):
    assert TokenOverlap(mode=mode, beta=beta).name == expected_name


def test_invalid_mode_and_beta():
    with pytest.raises(ValueError):
        TokenOverlap(mode="accuracy")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TokenOverlap(mode="fbeta", beta=0)


@pytest.mark.asyncio
async def test_non_string_input_scores_zero():
    result = await TokenOverlap().ascore(reference=None, response="x")  # type: ignore[arg-type]
    assert result.value == 0.0


def test_matches_prototype_numbers():
    """Same numbers as local_files/rag_prompt_advisor/scripts/score_and_check.py."""
    metrics = {
        "token_precision": TokenOverlap(mode="precision"),
        "token_recall": TokenOverlap(mode="recall"),
        "token_f1": TokenOverlap(mode="f1"),
        "token_f0_5": TokenOverlap(mode="fbeta", beta=0.5),
        "exact_match": ExactMatch(normalize=True),
    }
    means = {
        name: statistics.fmean(
            m.score(reference=ref, response=resp).value
            for ref, resp in PROTOTYPE_VAL_ROWS
        )
        for name, m in metrics.items()
    }
    assert means["token_precision"] == pytest.approx(0.5804, abs=1e-4)
    assert means["token_recall"] == pytest.approx(0.6857, abs=1e-4)
    assert means["token_f1"] == pytest.approx(0.4462, abs=1e-4)
    assert means["token_f0_5"] == pytest.approx(0.4592, abs=1e-4)
    assert means["exact_match"] == pytest.approx(0.25)


def test_exact_match_normalize():
    lenient = ExactMatch(normalize=True)
    assert (
        lenient.score(
            reference="Within 30 days of the purchase date",
            response="within 30 days of THE purchase date.",
        ).value
        == 1.0
    )
    assert lenient.score(reference="A VP", response="a vp").value == 1.0
    assert lenient.score(reference="A VP", response="A manager").value == 0.0


def test_exact_match_default_unchanged():
    strict = ExactMatch()
    assert strict.normalize is False
    assert strict.score(reference="A VP", response="A VP").value == 1.0
    assert strict.score(reference="A VP", response="A VP.").value == 0.0
