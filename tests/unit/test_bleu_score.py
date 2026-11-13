"""Tests for BleuScore (legacy and collections implementations).

These cover a class of silent scoring bugs: sacrebleu's ``corpus_bleu`` takes
``references`` as a list of reference *streams* (each stream holding one segment
per hypothesis), not as a list of segments. Passing per-sentence segments there
misaligns the streams, which makes sacrebleu score only the first sentence and
return a plausible-looking but wrong number instead of raising.
"""

import inspect

import pytest

try:
    import sacrebleu
    from sacrebleu import corpus_bleu
    from sacrebleu.metrics import BLEU
except ImportError:
    pytest.skip("sacrebleu not available", allow_module_level=True)

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore as LegacyBleuScore
from ragas.metrics.base import MetricType
from ragas.metrics.collections import BleuScore


def reference_bleu(reference: str, response: str) -> float:
    """Score the sample as a single segment -- the sacrebleu-correct baseline."""
    return corpus_bleu([response], [[reference]]).score / 100


async def score_legacy(reference: str, response: str) -> float:
    metric = LegacyBleuScore()
    sample = SingleTurnSample(reference=reference, response=response)
    return await metric._single_turn_ascore(sample, None)


async def score_v2(reference: str, response: str) -> float:
    result = await BleuScore().ascore(reference=reference, response=response)
    return result.value


# sacrebleu's BLEU settings that the hardcoded expectations below depend on.
#
# The expected scores in MULTI_SENTENCE_CASES -- and the documented-example
# value in TestBleuScoreEdgeCases -- are exact sacrebleu outputs rather than
# hand-derived numbers, generated with ``corpus_bleu([response], [[reference]])``
# on sacrebleu 2.6.0 with stock settings. They are therefore only stable while
# sacrebleu's defaults are unchanged: a release that switched the default
# tokenizer or flipped ``effective_order`` would move every one of them.
#
# ``TestSacrebleuAssumptions`` pins these so that such a release fails there,
# naming sacrebleu as the cause, instead of surfacing as several opaque float
# mismatches that read like a scoring regression in ragas. Pinning the defaults
# rather than ``sacrebleu.__version__`` keeps routine upgrades green -- only a
# behavioural change trips it.
SACREBLEU_BLEU_DEFAULTS = {
    "tokenizer": "13a",
    "smooth_method": "exp",
    "effective_order": False,
    "max_ngram_order": 4,
}


# Cases where the sentence-splitting bug silently produced a wrong score.
# Each is (description, reference, response, expected_score).
MULTI_SENTENCE_CASES = [
    (
        "identical multi-sentence text scores 1.0",
        "A is one. B is two. C is three.",
        "A is one. B is two. C is three.",
        1.0,
    ),
    (
        "garbage after the first sentence is penalised",
        "The cat sat on the mat quietly. The dog barked at the mailman.",
        "The cat sat on the mat quietly. zzz qqq wombat banana telephone.",
        0.4931,
    ),
    (
        "content missing after the first sentence is penalised",
        "The cat sat on the mat quietly. The dog barked at the mailman. Birds flew over the house.",
        "The cat sat on the mat quietly.",
        0.1969,
    ),
    (
        "a mismatched trailing sentence lowers the score",
        "The cat sat on the mat quietly. Alpha beta gamma delta epsilon.",
        "The cat sat on the mat quietly. Completely different words entirely here.",
        0.5296,
    ),
]


class TestSacrebleuAssumptions:
    """Pin the sacrebleu behaviour the hardcoded expectations rely on."""

    def test_bleu_defaults_unchanged(self):
        params = inspect.signature(BLEU.__init__).parameters
        actual = {
            "tokenizer": BLEU.TOKENIZER_DEFAULT,
            "smooth_method": params["smooth_method"].default,
            "effective_order": params["effective_order"].default,
            "max_ngram_order": params["max_ngram_order"].default,
        }
        assert actual == SACREBLEU_BLEU_DEFAULTS, (
            f"sacrebleu {sacrebleu.__version__} changed its BLEU defaults: "
            f"expected {SACREBLEU_BLEU_DEFAULTS}, got {actual}. The hardcoded "
            "scores in MULTI_SENTENCE_CASES and the documented-example value "
            "were generated under the previous defaults and need regenerating. "
            "This is a sacrebleu behaviour change, not a ragas scoring regression."
        )


class TestBleuScoreInit:
    """Initialization behaviour."""

    def test_legacy_defaults(self):
        metric = LegacyBleuScore()
        assert metric.name == "bleu_score"
        assert metric._required_columns == {
            MetricType.SINGLE_TURN: {"reference", "response"}
        }

    def test_v2_defaults(self):
        metric = BleuScore()
        assert metric.name == "bleu_score"
        assert metric.kwargs == {}

    def test_v2_custom_name_and_kwargs(self):
        metric = BleuScore(name="custom_bleu", kwargs={"smooth_method": "exp"})
        assert metric.name == "custom_bleu"
        assert metric.kwargs == {"smooth_method": "exp"}


class TestBleuScoreMultiSentence:
    """Regression tests for the reference-stream misalignment bug."""

    @pytest.mark.parametrize(
        "description,reference,response,expected",
        MULTI_SENTENCE_CASES,
        ids=[c[0] for c in MULTI_SENTENCE_CASES],
    )
    @pytest.mark.asyncio
    async def test_legacy_matches_expected(
        self, description, reference, response, expected
    ):
        assert await score_legacy(reference, response) == pytest.approx(
            expected, abs=1e-4
        )

    @pytest.mark.parametrize(
        "description,reference,response,expected",
        MULTI_SENTENCE_CASES,
        ids=[c[0] for c in MULTI_SENTENCE_CASES],
    )
    @pytest.mark.asyncio
    async def test_v2_matches_expected(
        self, description, reference, response, expected
    ):
        assert await score_v2(reference, response) == pytest.approx(expected, abs=1e-4)

    @pytest.mark.asyncio
    async def test_trailing_sentences_are_not_ignored(self):
        """A response is scored on all of its sentences, not just the first."""
        reference = "The cat sat on the mat quietly. The dog barked at the mailman."
        good = "The cat sat on the mat quietly. The dog barked at the mailman."
        bad = "The cat sat on the mat quietly. zzz qqq wombat banana telephone."

        assert await score_legacy(reference, good) > await score_legacy(reference, bad)


class TestBleuScoreAgainstSacrebleu:
    """Differential tests against sacrebleu used directly."""

    @pytest.mark.parametrize(
        "reference,response",
        [
            ("The cat sat on the mat.", "The cat sat on the mat."),
            ("A is one. B is two. C is three.", "A is one. B is two. C is three."),
            (
                "The Eiffel Tower is located in Paris.",
                "The Eiffel Tower is located in India.",
            ),
            (
                "Python is a high-level programming language. It was created by Guido van Rossum.",
                "Python is a programming language. It was developed by Guido van Rossum.",
            ),
            ("One. Two. Three. Four. Five.", "One. Two."),
            ("Short ref.", "A considerably longer response than the reference is."),
        ],
    )
    @pytest.mark.asyncio
    async def test_legacy_agrees_with_sacrebleu(self, reference, response):
        assert await score_legacy(reference, response) == pytest.approx(
            reference_bleu(reference, response), abs=1e-6
        )

    @pytest.mark.asyncio
    async def test_legacy_and_v2_agree(self):
        for _, reference, response, _ in MULTI_SENTENCE_CASES:
            assert await score_legacy(reference, response) == pytest.approx(
                await score_v2(reference, response), abs=1e-9
            )


class TestBleuScoreEdgeCases:
    """Boundary conditions."""

    @pytest.mark.asyncio
    async def test_single_sentence_perfect_match(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert await score_legacy(text, text) == pytest.approx(1.0)
        assert await score_v2(text, text) == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_documented_example_value_is_stable(self):
        """The value published in the metrics docs must not drift."""
        score = await score_v2(
            "The Eiffel Tower is located in Paris.",
            "The Eiffel Tower is located in India.",
        )
        assert score == pytest.approx(0.7071067811865478, abs=1e-9)

    @pytest.mark.asyncio
    async def test_completely_unrelated_text_scores_low(self):
        reference, response = (
            "The cat sat on the mat.",
            "Quantum entanglement defies locality.",
        )
        score = await score_legacy(reference, response)
        assert score == pytest.approx(reference_bleu(reference, response), abs=1e-9)
        assert score < 0.1

    @pytest.mark.parametrize(
        "reference,response",
        [("", "Some response text"), ("Some reference text", ""), ("", "")],
    )
    @pytest.mark.asyncio
    async def test_empty_strings_do_not_raise(self, reference, response):
        score = await score_legacy(reference, response)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_kwargs_are_forwarded_to_sacrebleu(self):
        metric = LegacyBleuScore(kwargs={"smooth_method": "add-k", "smooth_value": 1})
        sample = SingleTurnSample(
            reference="One two three four five.", response="One two three four six."
        )
        smoothed = await metric._single_turn_ascore(sample, None)
        expected = (
            corpus_bleu(
                ["One two three four six."],
                [["One two three four five."]],
                smooth_method="add-k",
                smooth_value=1,
            ).score
            / 100
        )
        assert smoothed == pytest.approx(expected, abs=1e-9)
