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
from ragas.metrics.collections.agent_goal_accuracy import (
    AgentGoalAccuracy,
    AgentGoalAccuracyWithoutReference,
    AgentGoalAccuracyWithReference,
)
from ragas.metrics.collections.answer_accuracy import AnswerAccuracy
from ragas.metrics.collections.answer_correctness import AnswerCorrectness
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.collections.chrf_score import CHRFScore
from ragas.metrics.collections.context_entity_recall import ContextEntityRecall
from ragas.metrics.collections.context_precision import (
    ContextPrecision,
    ContextPrecisionWithoutReference,
    ContextPrecisionWithReference,
    ContextUtilization,
)
from ragas.metrics.collections.context_recall import ContextRecall
from ragas.metrics.collections.context_relevance import ContextRelevance
from ragas.metrics.collections.datacompy_score import DataCompyScore
from ragas.metrics.collections.domain_specific_rubrics import (
    DomainSpecificRubrics,
    RubricsScoreWithoutReference,
    RubricsScoreWithReference,
)
from ragas.metrics.collections.factual_correctness import FactualCorrectness
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.instance_specific_rubrics import InstanceSpecificRubrics
from ragas.metrics.collections.multi_modal_faithfulness import MultiModalFaithfulness
from ragas.metrics.collections.multi_modal_relevance import MultiModalRelevance
from ragas.metrics.collections.noise_sensitivity import NoiseSensitivity
from ragas.metrics.collections.quoted_spans import QuotedSpansAlignment
from ragas.metrics.collections.response_groundedness import ResponseGroundedness
from ragas.metrics.collections.sql_semantic_equivalence import SQLSemanticEquivalence
from ragas.metrics.collections.summary_score import SummaryScore
from ragas.metrics.collections.tool_call_accuracy import ToolCallAccuracy
from ragas.metrics.collections.tool_call_f1 import ToolCallF1
from ragas.metrics.collections.topic_adherence import TopicAdherence

__all__ = [
    "BaseMetric",  # Base class
    # RAG metrics
    "AnswerAccuracy",
    "AnswerCorrectness",
    "AnswerRelevancy",
    "BleuScore",
    "CHRFScore",
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
    "MultiModalFaithfulness",
    "MultiModalRelevance",
    "NoiseSensitivity",
    "NonLLMStringSimilarity",
    "QuotedSpansAlignment",
    "ResponseGroundedness",
    "RougeScore",
    "SemanticSimilarity",
    "StringPresence",
    "SummaryScore",
    # Agent & Tool metrics
    "AgentGoalAccuracy",
    "AgentGoalAccuracyWithReference",
    "AgentGoalAccuracyWithoutReference",
    "ToolCallAccuracy",
    "ToolCallF1",
    "TopicAdherence",
    # Rubric metrics
    "DomainSpecificRubrics",
    "InstanceSpecificRubrics",
    "RubricsScoreWithoutReference",
    "RubricsScoreWithReference",
    # SQL & Data metrics
    "DataCompyScore",
    "SQLSemanticEquivalence",
]
