"""Collections of metrics using modern component architecture."""

from ragas.metrics.collections._bleu_score import BleuScore
from ragas.metrics.collections._rouge_score import RougeScore
from ragas.metrics.collections._semantic_similarity import SemanticSimilarity
from ragas.metrics.collections._string import (
    DistanceMeasure,
    ExactMatch,
    NonLLMStringSimilarity,
    StringPresence,
)
from ragas.metrics.collections.answer_accuracy import AnswerAccuracy
from ragas.metrics.collections.answer_correctness import AnswerCorrectness
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.collections.context_entity_recall import ContextEntityRecall
from ragas.metrics.collections.context_precision import (
    ContextPrecision,
    ContextPrecisionWithoutReference,
    ContextPrecisionWithReference,
    ContextUtilization,
)
from ragas.metrics.collections.context_recall import ContextRecall
from ragas.metrics.collections.context_relevance import ContextRelevance
from ragas.metrics.collections.factual_correctness import FactualCorrectness
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.noise_sensitivity import NoiseSensitivity
from ragas.metrics.collections.response_groundedness import ResponseGroundedness
from ragas.metrics.collections.summary_score import SummaryScore

__all__ = [
    "BaseMetric",  # Base class
    "AnswerAccuracy",
    "AnswerCorrectness",
    "AnswerRelevancy",
    "BleuScore",
    "ContextEntityRecall",
    "ContextRecall",
    "ContextPrecision",
    "ContextPrecisionWithReference",
    "ContextPrecisionWithoutReference",
    "ContextRelevance",
    "ContextUtilization",
    "DistanceMeasure",
    "ExactMatch",
    "FactualCorrectness",
    "Faithfulness",
    "NoiseSensitivity",
    "NonLLMStringSimilarity",
    "ResponseGroundedness",
    "RougeScore",
    "SemanticSimilarity",
    "StringPresence",
    "SummaryScore",
]
