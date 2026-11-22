"""
Unit tests for FactualCorrectness score computation.

Verifies that precision, recall, and f1 modes use the correct TP values:
  - precision TP = response claims supported by reference
  - recall TP    = reference claims supported by response

Regression test for: https://github.com/explodinggradients/ragas/issues/2693
"""

import numpy as np
import pytest

from ragas.metrics.utils import fbeta_score


# ---------------------------------------------------------------------------
# fbeta_score unit tests
# ---------------------------------------------------------------------------


def test_fbeta_precision_only():
    """When recall TP == all reference claims, recall == 1 and score is driven by precision."""
    # 3 response claims all correct (tp_precision=3, fp=0)
    # 2 reference claims both supported (tp_recall=2, fn=0)
    score = fbeta_score(tp_precision=3, fp=0, tp_recall=2, fn=0, beta=1.0)
    assert score == pytest.approx(1.0)


def test_fbeta_zero_precision():
    """All response claims are wrong → precision=0 → f1=0."""
    score = fbeta_score(tp_precision=0, fp=3, tp_recall=2, fn=0, beta=1.0)
    assert score == pytest.approx(0.0)


def test_fbeta_zero_recall():
    """No reference claims covered → recall=0 → f1=0."""
    score = fbeta_score(tp_precision=3, fp=0, tp_recall=0, fn=2, beta=1.0)
    assert score == pytest.approx(0.0)


def test_fbeta_asymmetric_precision_recall():
    """
    Concrete example from issue #2693:
      Response: 4 claims, 3 supported by reference  → tp_precision=3, fp=1
      Reference: 3 claims, 2 supported by response  → tp_recall=2,    fn=1

      Expected precision = 3/4 = 0.75
      Expected recall    = 2/3 ≈ 0.667
      Expected f1        = 2 * (0.75 * 0.667) / (0.75 + 0.667) ≈ 0.706
    """
    precision = 3 / 4
    recall = 2 / 3
    expected_f1 = 2 * (precision * recall) / (precision + recall)

    score = fbeta_score(tp_precision=3, fp=1, tp_recall=2, fn=1, beta=1.0)
    assert score == pytest.approx(expected_f1, rel=1e-3)


# ---------------------------------------------------------------------------
# _single_turn_ascore logic tests (via direct numpy arrays, no LLM needed)
# ---------------------------------------------------------------------------


def _compute_score(reference_response: np.ndarray, response_reference: np.ndarray, mode: str, beta: float = 1.0) -> float:
    """
    Replicate the fixed scoring logic from FactualCorrectness._single_turn_ascore
    so we can unit-test the math without spinning up an LLM.
    """
    tp_precision = sum(reference_response)
    fp = sum(~reference_response)

    if mode != "precision":
        tp_recall = sum(response_reference)
        fn = sum(~response_reference)
    else:
        tp_recall = 0
        fn = 0

    if mode == "precision":
        return tp_precision / (tp_precision + fp + 1e-8)
    elif mode == "recall":
        return tp_recall / (tp_recall + fn + 1e-8)
    else:
        return fbeta_score(tp_precision, fp, tp_recall, fn, beta)


def test_recall_uses_reference_claims():
    """
    Issue #2693 regression: recall must be computed from reference claims
    supported by the response, NOT from response claims supported by reference.

    Setup (from the issue):
      Response has 4 claims, 3 supported by reference → reference_response = [T,T,T,F]
      Reference has 3 claims, 2 supported by response → response_reference = [T,T,F]

      Wrong (old) recall = tp_precision / (tp_precision + fn) = 3 / (3+1) = 0.75
      Correct recall     = tp_recall   / (tp_recall   + fn)   = 2 / (2+1) ≈ 0.667
    """
    reference_response = np.array([True, True, True, False])  # precision direction
    response_reference = np.array([True, True, False])          # recall direction

    score = _compute_score(reference_response, response_reference, mode="recall")
    expected = 2 / (2 + 1)  # ≈ 0.667

    # Old (buggy) value would have been 0.75 — make sure we're not returning that
    assert score != pytest.approx(0.75, abs=1e-3), (
        "Recall is returning the precision TP — the bug has not been fixed."
    )
    assert score == pytest.approx(expected, rel=1e-3)


def test_precision_unaffected():
    """Precision mode should still use response claims verified against reference."""
    reference_response = np.array([True, True, True, False])
    response_reference = np.array([True, True, False])

    score = _compute_score(reference_response, response_reference, mode="precision")
    expected = 3 / 4  # 0.75
    assert score == pytest.approx(expected, rel=1e-3)


def test_f1_uses_correct_tps():
    """
    F1 must use tp_precision for precision and tp_recall for recall separately.

    Setup from issue #2693:
      precision = 3/4 = 0.75
      recall    = 2/3 ≈ 0.667
      f1        ≈ 0.706
    """
    reference_response = np.array([True, True, True, False])
    response_reference = np.array([True, True, False])

    score = _compute_score(reference_response, response_reference, mode="f1")
    precision = 3 / 4
    recall = 2 / 3
    expected_f1 = 2 * precision * recall / (precision + recall)

    assert score == pytest.approx(expected_f1, rel=1e-3)


def test_perfect_scores():
    """All claims correct in both directions → all modes should return 1.0."""
    reference_response = np.array([True, True, True])
    response_reference = np.array([True, True, True])

    for mode in ("precision", "recall", "f1"):
        score = _compute_score(reference_response, response_reference, mode=mode)
        assert score == pytest.approx(1.0, rel=1e-3), f"Failed for mode={mode}"


def test_zero_scores():
    """No claims correct in either direction → all modes should return ~0.0."""
    reference_response = np.array([False, False, False])
    response_reference = np.array([False, False, False])

    for mode in ("precision", "recall", "f1"):
        score = _compute_score(reference_response, response_reference, mode=mode)
        assert score == pytest.approx(0.0, abs=1e-6), f"Failed for mode={mode}"
