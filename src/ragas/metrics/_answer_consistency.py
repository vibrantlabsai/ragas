from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import HuggingfaceEmbeddings
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    SingleTurnMetric,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks.base import Callbacks


logger = logging.getLogger(__name__)


@dataclass
class AnswerConsistency(MetricWithEmbeddings, SingleTurnMetric):
    """
    Measures the consistency of answers across multiple responses to query variations.
    
    This metric evaluates the robustness of a RAG system by measuring how semantically
    similar the responses are when the same question is asked in different ways
    (e.g., paraphrased queries). High consistency indicates a robust system that
    provides stable answers regardless of query phrasing.
    
    The metric computes pairwise semantic similarity between all responses in
    `multi_responses` and returns the average similarity score.
    
    Attributes
    ----------
    name : str
        The name of the metric. Default is "answer_consistency".
    _required_columns : Dict[MetricType, Set[str]]
        Required columns for evaluation. Needs "multi_responses" which should
        contain a list of responses to query variations.
    output_type : MetricOutputType
        The output type is CONTINUOUS, ranging from 0.0 to 1.0.
    min_responses : int
        Minimum number of responses required in multi_responses. Default is 2.
        
    Notes
    -----
    - Score of 1.0 indicates perfect consistency (all responses are semantically identical)
    - Score of 0.0 indicates no consistency (responses are completely different)
    - Requires at least 2 responses in multi_responses to compute consistency
    
    Examples
    --------
    >>> from ragas.metrics import AnswerConsistency
    >>> from ragas.embeddings import OpenAIEmbeddings
    >>> 
    >>> metric = AnswerConsistency(embeddings=OpenAIEmbeddings())
    >>> sample = SingleTurnSample(
    ...     multi_responses=[
    ...         "Paris is the capital of France.",
    ...         "The capital city of France is Paris.",
    ...         "France's capital is Paris."
    ...     ]
    ... )
    >>> score = metric.score(sample)
    """

    name: str = "answer_consistency"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"multi_responses"}}
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS
    min_responses: int = 2

    def __post_init__(self):
        if self.min_responses < 2:
            raise ValueError(
                f"min_responses must be at least 2, got {self.min_responses}"
            )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        Compute the consistency score for multiple responses.
        
        Parameters
        ----------
        row : Dict
            Dictionary containing the sample data with "multi_responses" key.
        callbacks : Callbacks
            Callbacks for the evaluation process.
            
        Returns
        -------
        float
            Consistency score between 0.0 and 1.0.
            
        Raises
        ------
        ValueError
            If embeddings are not set or if multi_responses has fewer than min_responses.
        """
        assert self.embeddings is not None, (
            f"Error: '{self.name}' requires embeddings to be set."
        )

        if "multi_responses" not in row or not row["multi_responses"]:
            raise ValueError(
                "multi_responses is missing or empty in the test sample. "
                "Please provide at least 2 responses to query variations."
            )

        multi_responses = t.cast(t.List[str], row["multi_responses"])
        
        if len(multi_responses) < self.min_responses:
            raise ValueError(
                f"multi_responses must contain at least {self.min_responses} responses, "
                f"got {len(multi_responses)}"
            )

        # Handle empty strings by replacing with space
        multi_responses = [resp or " " for resp in multi_responses]

        # Get embeddings for all responses
        embeddings_list = []
        for response in multi_responses:
            # Handle both modern (BaseRagasEmbedding) and legacy (BaseRagasEmbeddings) interfaces
            if hasattr(self.embeddings, "aembed_text"):
                # Modern interface (BaseRagasEmbedding)
                embedding = np.array(await self.embeddings.aembed_text(response))  # type: ignore[attr-defined]
            else:
                # Legacy interface (BaseRagasEmbeddings)
                embedding = np.array(await self.embeddings.embed_text(response))  # type: ignore[misc]
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embeddings_list.append(embedding)

        # Compute pairwise cosine similarities
        similarities = []
        for emb1, emb2 in combinations(embeddings_list, 2):
            similarity = float(np.dot(emb1, emb2))
            # Clip to [0, 1] range (cosine similarity can be negative)
            similarity = max(0.0, min(1.0, similarity))
            similarities.append(similarity)

        # Return average pairwise similarity
        consistency_score = float(np.mean(similarities))
        return consistency_score


answer_consistency = AnswerConsistency()