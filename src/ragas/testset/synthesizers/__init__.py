import logging
import typing as t

from ragas.llms.base import BaseRagasLLM
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.synthesizers.multi_hop import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

from .base import BaseSynthesizer

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM

logger = logging.getLogger(__name__)

QueryDistribution = t.List[t.Tuple[BaseSynthesizer, float]]


def default_query_distribution(
    llm: t.Union[BaseRagasLLM, "InstructorBaseRagasLLM"],
    kg: t.Optional[KnowledgeGraph] = None,
    llm_context: t.Optional[str] = None,
) -> QueryDistribution:
    """ """
    default_queries = [
        SingleHopSpecificQuerySynthesizer(llm=llm, llm_context=llm_context),
        MultiHopAbstractQuerySynthesizer(llm=llm, llm_context=llm_context),
        MultiHopSpecificQuerySynthesizer(llm=llm, llm_context=llm_context),
    ]
    if kg is not None:
        available_queries = []
        for query in default_queries:
            try:
                if query.get_node_clusters(kg):
                    available_queries.append(query)
            except Exception as e:
                # Keep broad catch minimal for resilience; log and skip.
                logger.warning(
                    "Skipping %s due to unexpected error: %s",
                    getattr(query, "name", type(query).__name__),
                    e,
                )
                continue
        if not available_queries:
            raise ValueError(
                "No compatible query synthesizers for the provided KnowledgeGraph."
            )
    else:
        available_queries = default_queries

    return [(query, 1 / len(available_queries)) for query in available_queries]


__all__ = [
    "BaseSynthesizer",
    "default_query_distribution",
]
