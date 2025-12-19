import warnings

from ragas.metrics._answer_correctness import (
    AnswerCorrectness as _AnswerCorrectness,
    answer_correctness as _answer_correctness,
)
from ragas.metrics._answer_relevance import (
    AnswerRelevancy as _AnswerRelevancy,
    ResponseRelevancy as _ResponseRelevancy,
    answer_relevancy as _answer_relevancy,
)
from ragas.metrics._answer_similarity import (
    AnswerSimilarity as _AnswerSimilarity,
    SemanticSimilarity as _SemanticSimilarity,
    answer_similarity as _answer_similarity,
)
from ragas.metrics._aspect_critic import AspectCritic as _AspectCritic
from ragas.metrics._bleu_score import BleuScore as _BleuScore
from ragas.metrics._chrf_score import ChrfScore as _ChrfScore
from ragas.metrics._context_entities_recall import (
    ContextEntityRecall as _ContextEntityRecall,
    context_entity_recall as _context_entity_recall,
)
from ragas.metrics._context_precision import (
    ContextPrecision as _ContextPrecision,
    ContextUtilization as _ContextUtilization,
    IDBasedContextPrecision as _IDBasedContextPrecision,
    LLMContextPrecisionWithoutReference as _LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference as _LLMContextPrecisionWithReference,
    NonLLMContextPrecisionWithReference as _NonLLMContextPrecisionWithReference,
    context_precision as _context_precision,
)
from ragas.metrics._context_recall import (
    ContextRecall as _ContextRecall,
    IDBasedContextRecall as _IDBasedContextRecall,
    LLMContextRecall as _LLMContextRecall,
    NonLLMContextRecall as _NonLLMContextRecall,
    context_recall as _context_recall,
)
from ragas.metrics._datacompy_score import DataCompyScore as _DataCompyScore
from ragas.metrics._domain_specific_rubrics import RubricsScore as _RubricsScore
from ragas.metrics._factual_correctness import FactualCorrectness as _FactualCorrectness
from ragas.metrics._faithfulness import (
    Faithfulness as _Faithfulness,
    FaithfulnesswithHHEM as _FaithfulnesswithHHEM,
    faithfulness as _faithfulness,
)
from ragas.metrics._goal_accuracy import (
    AgentGoalAccuracyWithoutReference as _AgentGoalAccuracyWithoutReference,
    AgentGoalAccuracyWithReference as _AgentGoalAccuracyWithReference,
)
from ragas.metrics._instance_specific_rubrics import InstanceRubrics as _InstanceRubrics
from ragas.metrics._multi_modal_faithfulness import (
    MultiModalFaithfulness as _MultiModalFaithfulness,
    multimodal_faithness as _multimodal_faithness,
)
from ragas.metrics._multi_modal_relevance import (
    MultiModalRelevance as _MultiModalRelevance,
    multimodal_relevance as _multimodal_relevance,
)
from ragas.metrics._noise_sensitivity import NoiseSensitivity as _NoiseSensitivity
from ragas.metrics._nv_metrics import (
    AnswerAccuracy as _AnswerAccuracy,
    ContextRelevance as _ContextRelevance,
    ResponseGroundedness as _ResponseGroundedness,
)
from ragas.metrics._rouge_score import RougeScore as _RougeScore
from ragas.metrics._simple_criteria import SimpleCriteriaScore as _SimpleCriteriaScore
from ragas.metrics._sql_semantic_equivalence import (
    LLMSQLEquivalence as _LLMSQLEquivalence,
)
from ragas.metrics._string import (
    DistanceMeasure as _DistanceMeasure,
    ExactMatch as _ExactMatch,
    NonLLMStringSimilarity as _NonLLMStringSimilarity,
    StringPresence as _StringPresence,
)
from ragas.metrics._summarization import (
    SummarizationScore as _SummarizationScore,
    summarization_score as _summarization_score,
)
from ragas.metrics._tool_call_accuracy import ToolCallAccuracy as _ToolCallAccuracy
from ragas.metrics._tool_call_f1 import ToolCallF1 as _ToolCallF1
from ragas.metrics._topic_adherence import TopicAdherenceScore as _TopicAdherenceScore
from ragas.metrics.base import (
    Metric,
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    MultiTurnMetric,
    SimpleBaseMetric as BaseMetric,
    SimpleLLMMetric as LLMMetric,
    SingleTurnMetric,
)
from ragas.metrics.discrete import DiscreteMetric, discrete_metric
from ragas.metrics.numeric import NumericMetric, numeric_metric
from ragas.metrics.ranking import RankingMetric, ranking_metric
from ragas.metrics.result import MetricResult

__all__ = [
    # basic metrics primitives
    "Metric",
    "MetricType",
    "MetricWithEmbeddings",
    "MetricWithLLM",
    "SingleTurnMetric",
    "MultiTurnMetric",
    "MetricOutputType",
    # LLM-based metrics (moved from experimental)
    "BaseMetric",
    "LLMMetric",
    "MetricResult",
    "DiscreteMetric",
    "NumericMetric",
    "RankingMetric",
    "discrete_metric",
    "numeric_metric",
    "ranking_metric",
    # Note: Specific metric classes and instances are deprecated from this module
    # and should be imported from ragas.metrics.collections instead.
    # They remain accessible via __getattr__ for backwards compatibility.
]

# Mapping of deprecated metric names to their actual implementations
_DEPRECATED_METRICS = {
    # Specific metric classes and instances (deprecated, use ragas.metrics.collections)
    "AnswerAccuracy": _AnswerAccuracy,
    "AnswerCorrectness": _AnswerCorrectness,
    "answer_correctness": _answer_correctness,
    "AnswerRelevancy": _AnswerRelevancy,
    "answer_relevancy": _answer_relevancy,
    "AnswerSimilarity": _AnswerSimilarity,
    "answer_similarity": _answer_similarity,
    "AspectCritic": _AspectCritic,
    "BleuScore": _BleuScore,
    "ChrfScore": _ChrfScore,
    "ContextEntityRecall": _ContextEntityRecall,
    "context_entity_recall": _context_entity_recall,
    "ContextPrecision": _ContextPrecision,
    "context_precision": _context_precision,
    "ContextRecall": _ContextRecall,
    "context_recall": _context_recall,
    "ContextRelevance": _ContextRelevance,
    "ContextUtilization": _ContextUtilization,
    "DataCompyScore": _DataCompyScore,
    "DistanceMeasure": _DistanceMeasure,
    "ExactMatch": _ExactMatch,
    "FactualCorrectness": _FactualCorrectness,
    "Faithfulness": _Faithfulness,
    "faithfulness": _faithfulness,
    "FaithfulnesswithHHEM": _FaithfulnesswithHHEM,
    "IDBasedContextPrecision": _IDBasedContextPrecision,
    "IDBasedContextRecall": _IDBasedContextRecall,
    "InstanceRubrics": _InstanceRubrics,
    "LLMContextPrecisionWithoutReference": _LLMContextPrecisionWithoutReference,
    "LLMContextPrecisionWithReference": _LLMContextPrecisionWithReference,
    "LLMContextRecall": _LLMContextRecall,
    "LLMSQLEquivalence": _LLMSQLEquivalence,
    "MultiModalFaithfulness": _MultiModalFaithfulness,
    "multimodal_faithness": _multimodal_faithness,
    "MultiModalRelevance": _MultiModalRelevance,
    "multimodal_relevance": _multimodal_relevance,
    "NoiseSensitivity": _NoiseSensitivity,
    "NonLLMContextPrecisionWithReference": _NonLLMContextPrecisionWithReference,
    "NonLLMContextRecall": _NonLLMContextRecall,
    "NonLLMStringSimilarity": _NonLLMStringSimilarity,
    "ResponseGroundedness": _ResponseGroundedness,
    "ResponseRelevancy": _ResponseRelevancy,
    "RougeScore": _RougeScore,
    "RubricsScore": _RubricsScore,
    "SemanticSimilarity": _SemanticSimilarity,
    "SimpleCriteriaScore": _SimpleCriteriaScore,
    "StringPresence": _StringPresence,
    "SummarizationScore": _SummarizationScore,
    "summarization_score": _summarization_score,
    "ToolCallAccuracy": _ToolCallAccuracy,
    "ToolCallF1": _ToolCallF1,
    "TopicAdherenceScore": _TopicAdherenceScore,
    "AgentGoalAccuracyWithoutReference": _AgentGoalAccuracyWithoutReference,
    "AgentGoalAccuracyWithReference": _AgentGoalAccuracyWithReference,
}

_DEPRECATION_MESSAGE = (
    "Importing {name} from 'ragas.metrics' is deprecated and will be removed in v1.0. "
    "Please use 'ragas.metrics.collections' instead. "
    "Example: from ragas.metrics.collections import {name}"
)


def __getattr__(name: str):
    if name in _DEPRECATED_METRICS:
        warnings.warn(
            _DEPRECATION_MESSAGE.format(name=name),
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_METRICS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
