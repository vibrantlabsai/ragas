from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


class ResponseRelevanceOutput(BaseModel):
    question: str
    noncommittal: int


class ResponseRelevanceInput(BaseModel):
    response: str


class ResponseRelevancePrompt(
    PydanticPrompt[ResponseRelevanceInput, ResponseRelevanceOutput]
):
    instruction = """Generate a question for the given answer and Identify if answer is noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers"""
    input_model = ResponseRelevanceInput
    output_model = ResponseRelevanceOutput
    examples = [
        (
            ResponseRelevanceInput(
                response="""Albert Einstein was born in Germany.""",
            ),
            ResponseRelevanceOutput(
                question="Where was Albert Einstein born?",
                noncommittal=0,
            ),
        ),
        (
            ResponseRelevanceInput(
                response="""I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022. """,
            ),
            ResponseRelevanceOutput(
                question="What was the groundbreaking feature of the smartphone invented in 2023?",
                noncommittal=1,
            ),
        ),
    ]


@dataclass
class ResponseRelevancy(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    Attributes
    ----------
    name: string
        The name of the metrics
    strictness: int
        Here indicates the number questions generated per answer.
        Ideal range between 3 to 5.
    embeddings: Embedding
        The langchain wrapper of Embedding object.
        E.g. HuggingFaceEmbeddings('BAAI/bge-base-en')
    """

    name: str = "answer_relevancy"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
            }
        }
    )
    output_type = MetricOutputType.CONTINUOUS

    question_generation: PydanticPrompt = ResponseRelevancePrompt()
    strictness: int = 3

    def calculate_similarity(self, question: str, generated_questions: list[str]):
        assert self.embeddings is not None, (
            f"Error: '{self.name}' requires embeddings to be set."
        )
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)  # type: ignore[attr-defined]
        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions)  # type: ignore[attr-defined]
        ).reshape(len(generated_questions), -1)
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )

    def _calculate_score(
        self, answers: t.Sequence[ResponseRelevanceOutput], row: t.Dict
    ) -> float:
        question = row["user_input"]
        gen_questions = [answer.question for answer in answers]
        all_noncommittal = np.all([answer.noncommittal for answer in answers])
        
        # Store for logging (only if score is 0.0 or NaN)
        question_preview = question[:200] if question else ""
        answer_preview = row.get("response", "")[:200] if row.get("response") else ""
        
        # Initialize calculation variables (will be set if not NaN)
        cosine_sim = None
        mean_similarity = None
        noncommittal_multiplier = None
        is_nan_reason = None
        
        if all(q == "" for q in gen_questions):
            is_nan_reason = "Invalid JSON response - all generated questions are empty"
            logger.warning(
                "Invalid JSON response. Expected dictionary with key 'question'"
            )
            score = np.nan
        else:
            # CALCULATION: Cosine similarity
            cosine_sim = self.calculate_similarity(question, gen_questions)
            mean_similarity = cosine_sim.mean()
            noncommittal_multiplier = int(not all_noncommittal)
            score = mean_similarity * noncommittal_multiplier

        # DETAILED LOGGING ONLY WHEN SCORE IS 0.0 OR NaN
        if score == 0.0 or (isinstance(score, float) and np.isnan(score)):
            # INPUT LOGGING
            logger.warning(f"[ANSWER_RELEVANCY INPUT] Question (first 200 chars): {question_preview}")
            logger.warning(f"[ANSWER_RELEVANCY INPUT] Answer (first 200 chars): {answer_preview}")
            logger.warning(f"[ANSWER_RELEVANCY INPUT] Generated questions count: {len(gen_questions)}")
            
            # CALCULATION LOGGING
            for i, (gen_q, answer) in enumerate(zip(gen_questions, answers)):
                gen_q_preview = gen_q[:150] if gen_q else ""
                logger.warning(f"[ANSWER_RELEVANCY CALCULATION] Generated question {i+1} (first 150 chars): {gen_q_preview}")
                logger.warning(f"[ANSWER_RELEVANCY CALCULATION] Noncommittal {i+1}: {answer.noncommittal}")
            
            logger.warning(f"[ANSWER_RELEVANCY CALCULATION] All noncommittal: {all_noncommittal}")
            
            if np.isnan(score):
                # NaN case: log the reason
                logger.warning(f"[ANSWER_RELEVANCY CALCULATION] Score is NaN - Reason: {is_nan_reason}")
                logger.warning(f"[ANSWER_RELEVANCY CALCULATION] Generated questions were: {gen_questions}")
            else:
                # 0.0 case: log calculation details
                logger.warning(f"[ANSWER_RELEVANCY CALCULATION] Cosine similarities: {cosine_sim}")
                logger.warning(f"[ANSWER_RELEVANCY CALCULATION] Mean similarity: {mean_similarity}")
                logger.warning(f"[ANSWER_RELEVANCY CALCULATION] Noncommittal multiplier: {noncommittal_multiplier}")
                logger.warning(f"[ANSWER_RELEVANCY CALCULATION] Final score calculation: {mean_similarity} * {noncommittal_multiplier} = {score}")
            
            # OUTPUT LOGGING
            logger.warning(f"[ANSWER_RELEVANCY OUTPUT] Score is {score}")
            if np.isnan(score):
                logger.warning(f"[ANSWER_RELEVANCY OUTPUT] NaN reason: {is_nan_reason}")
            logger.warning(f"[ANSWER_RELEVANCY OUTPUT] Question: {question_preview}")
            logger.warning(f"[ANSWER_RELEVANCY OUTPUT] Generated questions: {gen_questions}")
            logger.warning(f"[ANSWER_RELEVANCY OUTPUT] All noncommittal: {all_noncommittal}")
            if not np.isnan(score) and mean_similarity is not None:
                logger.warning(f"[ANSWER_RELEVANCY OUTPUT] Mean similarity was: {mean_similarity}")

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt_input = ResponseRelevanceInput(response=row["response"])

        responses = await self.question_generation.generate_multiple(
            data=prompt_input, llm=self.llm, callbacks=callbacks, n=self.strictness
        )

        return self._calculate_score(responses, row)


class AnswerRelevancy(ResponseRelevancy):
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


answer_relevancy = AnswerRelevancy()
