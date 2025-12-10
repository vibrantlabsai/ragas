from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._string import NonLLMStringSimilarity
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
    ensembler,
)
from ragas.prompt import PydanticPrompt
from ragas.run_config import RunConfig
from ragas.utils import deprecated

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class QAC(BaseModel):
    question: str = Field(..., description="Question")
    context: str = Field(..., description="Context")
    answer: str = Field(..., description="Answer")


class Verification(BaseModel):
    reason: str = Field(..., description="Reason for verification")
    verdict: int = Field(..., description="Binary (0/1) verdict of verification")


class ContextPrecisionPrompt(PydanticPrompt[QAC, Verification]):
    name: str = "context_precision"
    instruction: str = 'Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not with json output.'
    input_model = QAC
    output_model = Verification
    examples = [
        (
            QAC(
                question="What can you tell me about Albert Einstein?",
                context="Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.",
                answer="Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics.",
            ),
            Verification(
                reason="The provided context was indeed useful in arriving at the given answer. The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer.",
                verdict=1,
            ),
        ),
        (
            QAC(
                question="who won 2020 icc world cup?",
                context="The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.",
                answer="England",
            ),
            Verification(
                reason="the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.",
                verdict=1,
            ),
        ),
        (
            QAC(
                question="What is the tallest mountain in the world?",
                context="The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.",
                answer="Mount Everest.",
            ),
            Verification(
                reason="the provided context discusses the Andes mountain range, which, while impressive, does not include Mount Everest or directly relate to the question about the world's tallest mountain.",
                verdict=0,
            ),
        ),
    ]


@dataclass
class LLMContextPrecisionWithReference(MetricWithLLM, SingleTurnMetric):
    """
    Average Precision is a metric that evaluates whether all of the
    relevant items selected by the model are ranked higher or not.

    Attributes
    ----------
    name : str
    evaluation_mode: EvaluationMode
    context_precision_prompt: Prompt
    """

    name: str = "llm_context_precision_with_reference"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "retrieved_contexts",
                "reference",
            }
        }
    )
    output_type = MetricOutputType.CONTINUOUS
    context_precision_prompt: PydanticPrompt = field(
        default_factory=ContextPrecisionPrompt
    )
    max_retries: int = 1

    def _get_row_attributes(self, row: t.Dict) -> t.Tuple[str, t.List[str], t.Any]:
        return row["user_input"], row["retrieved_contexts"], row["reference"]

    def _calculate_average_precision(
        self, verifications: t.List[Verification]
    ) -> float:
        score = np.nan
        is_nan_reason = None

        try:
            verdict_list = [1 if ver.verdict else 0 for ver in verifications]
            denominator = sum(verdict_list) + 1e-10
            
            # CALCULATION: Numerator calculation step-by-step
            numerator_terms = []
            for i in range(len(verdict_list)):
                prefix_sum = sum(verdict_list[: i + 1])
                term = (prefix_sum / (i + 1)) * verdict_list[i]
                numerator_terms.append(term)
            
            numerator = sum(numerator_terms)
            score = numerator / denominator
            
            # Only log calculation details if score is 0.0 or NaN (will be checked by caller)
            if score == 0.0 or (isinstance(score, float) and np.isnan(score)):
                logger.warning(f"[CONTEXT_PRECISION CALCULATION] Verdict list: {verdict_list}")
                logger.warning(f"[CONTEXT_PRECISION CALCULATION] Denominator: {denominator}")
                logger.warning(f"[CONTEXT_PRECISION CALCULATION] Numerator: {numerator}")
                if np.isnan(score):
                    logger.warning(f"[CONTEXT_PRECISION CALCULATION] Score is NaN - calculation resulted in NaN")
                else:
                    logger.warning(f"[CONTEXT_PRECISION CALCULATION] Score: {numerator} / {denominator} = {score}")
                    for i in range(len(verdict_list)):
                        prefix_sum = sum(verdict_list[: i + 1])
                        term = (prefix_sum / (i + 1)) * verdict_list[i]
                        logger.warning(f"[CONTEXT_PRECISION CALCULATION] Term {i+1}: prefix_sum={prefix_sum}, divisor={i+1}, verdict={verdict_list[i]}, term={term}")
        except Exception as e:
            is_nan_reason = f"Exception in calculation: {e}"
            logger.warning(
                "Invalid response format. Expected a list of dictionaries with keys 'verdict'"
            )
            score = np.nan
        
        if np.isnan(score) and is_nan_reason:
            logger.warning(f"[CONTEXT_PRECISION CALCULATION] NaN reason: {is_nan_reason}")
        
        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(
        self,
        row: t.Dict,
        callbacks: Callbacks,
    ) -> float:
        assert self.llm is not None, "LLM is not set"

        user_input, retrieved_contexts, reference = self._get_row_attributes(row)
        
        # Store for logging (only if score is 0.0)
        question_preview = user_input[:200] if user_input else ""
        answer_preview = reference[:200] if reference else ""
        contexts_count = len(retrieved_contexts) if retrieved_contexts else 0
        contexts_preview = [ctx[:200] for ctx in retrieved_contexts[:3]] if retrieved_contexts else []
        
        responses = []
        for i, context in enumerate(retrieved_contexts):
            verdicts: t.List[
                Verification
            ] = await self.context_precision_prompt.generate_multiple(
                data=QAC(
                    question=user_input,
                    context=context,
                    answer=reference,
                ),
                llm=self.llm,
                callbacks=callbacks,
            )

            responses.append([result.model_dump() for result in verdicts])

        answers = []
        for i, response in enumerate(responses):
            agg_answer = ensembler.from_discrete([response], "verdict")
            answers.append(Verification(**agg_answer[0]))

        score = self._calculate_average_precision(answers)
        
        # DETAILED LOGGING ONLY WHEN SCORE IS 0.0 OR NaN
        if score == 0.0 or (isinstance(score, float) and np.isnan(score)):
            # INPUT LOGGING
            logger.warning(f"[CONTEXT_PRECISION INPUT] Question (first 200 chars): {question_preview}")
            logger.warning(f"[CONTEXT_PRECISION INPUT] Answer (first 200 chars): {answer_preview}")
            logger.warning(f"[CONTEXT_PRECISION INPUT] Contexts count: {contexts_count}")
            if contexts_preview:
                for i, ctx_preview in enumerate(contexts_preview):
                    logger.warning(f"[CONTEXT_PRECISION INPUT] Context {i+1} (first 200 chars): {ctx_preview}")
            
            # CALCULATION LOGGING
            for i, context in enumerate(retrieved_contexts):
                context_preview = context[:200] if context else ""
                logger.warning(f"[CONTEXT_PRECISION CALCULATION] Processing context {i+1}/{contexts_count}: {context_preview}...")
                
                # Re-extract verdicts for logging (we already have them in responses)
                if i < len(responses):
                    verdicts_data = responses[i]
                    logger.warning(f"[CONTEXT_PRECISION CALCULATION] Context {i+1} verdicts count: {len(verdicts_data)}")
                    for j, verdict_data in enumerate(verdicts_data):
                        reason_preview = verdict_data.get('reason', '')[:150] if verdict_data.get('reason') else ""
                        verdict_value = verdict_data.get('verdict', 'N/A')
                        logger.warning(f"[CONTEXT_PRECISION CALCULATION] Context {i+1}, Verdict {j+1}: verdict={verdict_value}, reason='{reason_preview}...'")
            
            for i, answer in enumerate(answers):
                logger.warning(f"[CONTEXT_PRECISION CALCULATION] Context {i+1} aggregated verdict: {answer.verdict}")
            
            # OUTPUT LOGGING
            logger.warning(f"[CONTEXT_PRECISION OUTPUT] Score is {score}")
            if np.isnan(score):
                logger.warning(f"[CONTEXT_PRECISION OUTPUT] NaN reason: Invalid response format or calculation error")
            logger.warning(f"[CONTEXT_PRECISION OUTPUT] Question: {question_preview}")
            logger.warning(f"[CONTEXT_PRECISION OUTPUT] Answer: {answer_preview}")
            logger.warning(f"[CONTEXT_PRECISION OUTPUT] Contexts count: {contexts_count}")
            logger.warning(f"[CONTEXT_PRECISION OUTPUT] All verdicts: {[ans.verdict for ans in answers]}")
        
        return score


@dataclass
class LLMContextPrecisionWithoutReference(LLMContextPrecisionWithReference):
    name: str = "llm_context_precision_without_reference"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )

    def _get_row_attributes(self, row: t.Dict) -> t.Tuple[str, t.List[str], t.Any]:
        return row["user_input"], row["retrieved_contexts"], row["response"]


@dataclass
class NonLLMContextPrecisionWithReference(SingleTurnMetric):
    name: str = "non_llm_context_precision_with_reference"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_contexts",
                "reference_contexts",
            }
        }
    )
    distance_measure: SingleTurnMetric = field(
        default_factory=lambda: NonLLMStringSimilarity()
    )
    threshold: float = 0.5

    def __post_init__(self):
        if isinstance(self.distance_measure, MetricWithLLM):
            raise ValueError(
                "distance_measure must not be an instance of MetricWithLLM for NonLLMContextPrecisionWithReference"
            )

    def init(self, run_config: RunConfig) -> None: ...

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        sample = SingleTurnSample(**row)
        return await self._single_turn_ascore(sample, callbacks)

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved_contexts = sample.retrieved_contexts
        reference_contexts = sample.reference_contexts
        assert retrieved_contexts is not None, "retrieved_contexts is empty"
        assert reference_contexts is not None, "reference_contexts is empty"

        scores = []
        for rc in retrieved_contexts:
            scores.append(
                max(
                    [
                        await self.distance_measure.single_turn_ascore(
                            SingleTurnSample(reference=rc, response=ref), callbacks
                        )
                        for ref in reference_contexts
                    ]
                )
            )
        scores = [1 if score >= self.threshold else 0 for score in scores]
        return self._calculate_average_precision(scores)

    def _calculate_average_precision(self, verdict_list: t.List[int]) -> float:
        score = np.nan

        denominator = sum(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        score = numerator / denominator
        return score


@dataclass
class IDBasedContextPrecision(SingleTurnMetric):
    """
    Calculates context precision by directly comparing retrieved context IDs with reference context IDs.
    The score represents what proportion of the retrieved context IDs are actually relevant (present in reference).

    This metric works with both string and integer IDs.

    Attributes
    ----------
    name : str
        Name of the metric
    """

    name: str = "id_based_context_precision"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_context_ids",
                "reference_context_ids",
            }
        }
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS

    def init(self, run_config: RunConfig) -> None: ...

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved_context_ids = sample.retrieved_context_ids
        reference_context_ids = sample.reference_context_ids
        assert retrieved_context_ids is not None, "retrieved_context_ids is empty"
        assert reference_context_ids is not None, "reference_context_ids is empty"

        # Convert all IDs to strings to ensure consistent comparison
        retrieved_ids_set = set(str(id) for id in retrieved_context_ids)
        reference_ids_set = set(str(id) for id in reference_context_ids)

        # Calculate precision score
        total_retrieved = len(retrieved_ids_set)
        if total_retrieved == 0:
            logger.warning(
                "No retrieved context IDs provided, cannot calculate precision."
            )
            return np.nan

        # Count how many retrieved IDs match reference IDs
        hits = sum(
            1 for ret_id in retrieved_ids_set if str(ret_id) in reference_ids_set
        )

        # For precision, we calculate: relevant retrieved / total retrieved
        score = hits / total_retrieved
        return score

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


@dataclass
class ContextPrecision(LLMContextPrecisionWithReference):
    name: str = "context_precision"

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        return await super()._single_turn_ascore(sample, callbacks)

    @deprecated(
        since="0.2", removal="0.3", alternative="LLMContextPrecisionWithReference"
    )
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


@dataclass
class ContextUtilization(LLMContextPrecisionWithoutReference):
    name: str = "context_utilization"

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        return await super()._single_turn_ascore(sample, callbacks)

    @deprecated(
        since="0.2", removal="0.3", alternative="LLMContextPrecisionWithoutReference"
    )
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


context_precision = ContextPrecision()
context_utilization = ContextUtilization()
