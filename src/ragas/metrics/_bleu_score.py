import typing as t
from dataclasses import dataclass, field

from langchain_core.callbacks import Callbacks

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricType, SingleTurnMetric
from ragas.run_config import RunConfig


@dataclass
class BleuScore(SingleTurnMetric):
    name: str = "bleu_score"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"reference", "response"}}
    )
    kwargs: t.Dict[str, t.Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            from sacrebleu import corpus_bleu
        except ImportError:
            raise ImportError(
                "sacrebleu is required for bleu score. Please install it using `pip install sacrebleu`"
            )
        self.corpus_bleu = corpus_bleu

    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        reference, response = sample.reference, sample.response
        assert isinstance(reference, str), "BleuScore expects a valid reference string"
        assert isinstance(response, str), "BleuScore expects a valid response string"

        # Pass full texts as a single segment. Splitting into sentences and
        # building N reference streams of length 1 made sacrebleu's zip truncate
        # after the first sentence (only the first sentence was scored).
        score = self.corpus_bleu([response], [[reference]], **self.kwargs).score / 100
        assert isinstance(score, float), "Expecting a float"
        return score

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)
