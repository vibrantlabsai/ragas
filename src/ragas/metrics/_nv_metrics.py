from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.callbacks import Callbacks
from langchain_core.prompt_values import StringPromptValue

from ragas.dataset_schema import SingleTurnSample
from ragas.llms.base import BaseRagasLLM
from ragas.metrics.base import MetricType, MetricWithLLM, SingleTurnMetric

logger = logging.getLogger(__name__)


@dataclass
class AnswerAccuracy(MetricWithLLM, SingleTurnMetric):
    """
    Measures answer accuracy compared to ground truth given a user_input.
    This metric averages two distinct judge prompts to evaluate.

    Top10, Zero-shoot LLM-as-a-Judge Leaderboard:
    1)- nvidia/Llama-3_3-Nemotron-Super-49B-v1
    2)- mistralai/mixtral-8x22b-instruct-v0.1
    3)- mistralai/mixtral-8x7b-instruct-v0.1
    4)- meta/llama-3.1-70b-instruct
    5)- meta/llama-3.3-70b-instruct
    6)- meta/llama-3.1-405b-instruct
    7)- mistralai/mistral-nemo-12b-instruct
    8)- nvidia/llama-3.1-nemotron-70b-instruct
    9)- meta/llama-3.1-8b-instruct
    10)- google/gemma-2-2b-it
    The top1 LB model have high correlation with human judges (~0.92).

    Attributes
    ----------
    name: string
        The name of the metrics

    answer_accuracy:
        The AnswerAccuracy object
    """

    name: str = field(default="nv_accuracy", repr=True)  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
                "reference",
            },
        }
    )
    template_accuracy1 = (
        "Instruction: You are a world class state of the art assistant for rating "
        "a User Answer given a Question. The Question is completely answered by the Reference Answer.\n"
        "Say 4, if User Answer is full contained and equivalent to Reference Answer"
        "in all terms, topics, numbers, metrics, dates and units.\n"
        "Say 2, if User Answer is partially contained and almost equivalent to Reference Answer"
        "in all terms, topics, numbers, metrics, dates and units.\n"
        "Say 0, if User Answer is not contained in Reference Answer or not accurate in all terms, topics,"
        "numbers, metrics, dates and units or the User Answer do not answer the question.\n"
        "Do not explain or justify your rating. Your rating must be only 4, 2 or 0 according to the instructions above.\n"
        "### Question: {query}\n"
        "### {answer0}: {sentence_inference}\n"
        "### {answer1}: {sentence_true}\n"
        "The rating is:\n"
    )
    template_accuracy2 = (
        "I will rate the User Answer in comparison to the Reference Answer for a given Question.\n"
        "A rating of 4 indicates that the User Answer is entirely consistent with the Reference Answer, covering all aspects, topics, numbers, metrics, dates, and units.\n"
        "A rating of 2 signifies that the User Answer is mostly aligned with the Reference Answer, with minor discrepancies in some areas.\n"
        "A rating of 0 means that the User Answer is either inaccurate, incomplete, or unrelated to the Reference Answer, or it fails to address the Question.\n"
        "I will provide the rating without any explanation or justification, adhering to the following scale: 0 (no match), 2 (partial match), 4 (exact match).\n"
        "Do not explain or justify my rating. My rating must be only 4, 2 or 0 only.\n\n"
        "Question: {query}\n\n"
        "{answer0}: {sentence_inference}\n\n"
        "{answer1}: {sentence_true}\n\n"
        "Rating: "
    )
    retry = 5  # Number of retries if rating is not in the first 8 tokens.

    def process_score(self, response):
        for i in range(5):
            if str(i) in response[:]:
                return i / 4
        return np.nan

    def average_scores(self, score0, score1):
        score = np.nan
        if score0 >= 0 and score1 >= 0:
            score = (score0 + score1) / 2
        elif score0 >= 0:
            score = score0
        elif score1 >= 0:
            score = score1
        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not set"
        assert sample.user_input is not None, "User input is not set"
        assert sample.response is not None, "Response is not set"
        assert sample.reference is not None, "Reference is not set"

        try:
            score_ref_gen = score_gen_ref = np.nan
            for retry in range(self.retry):
                formatted_prompt = StringPromptValue(
                    text=self.template_accuracy1.format(
                        query=sample.user_input,
                        answer0="User Answer",
                        answer1="Reference Answer",
                        sentence_inference=sample.response,
                        sentence_true=sample.reference,
                    )
                )
                req0 = t.cast(BaseRagasLLM, self.llm).agenerate_text(
                    formatted_prompt,
                    n=1,
                    temperature=0.10,
                )
                resp0 = await req0
                score_ref_gen = resp0.generations[0][0].text
                score_ref_gen = self.process_score(score_ref_gen)
                if score_ref_gen == score_ref_gen:
                    break
                else:
                    logger.warning(f"Retry: {retry}")

            for retry in range(self.retry):
                formatted_prompt = StringPromptValue(
                    text=self.template_accuracy2.format(
                        query=sample.user_input,
                        answer0="Reference Answer",
                        answer1="User Answer",
                        sentence_inference=sample.reference,
                        sentence_true=sample.response,
                    )
                )
                req1 = t.cast(BaseRagasLLM, self.llm).agenerate_text(
                    formatted_prompt,
                    n=1,
                    temperature=0.10,
                )
                resp1 = await req1
                score_gen_ref = resp1.generations[0][0].text
                score_gen_ref = self.process_score(score_gen_ref)
                if score_gen_ref == score_gen_ref:
                    break
                else:
                    logger.warning(f"Retry: {retry}")

            score = self.average_scores(score_ref_gen, score_gen_ref)

        except Exception as e:
            logger.warning(
                f"An error occurred: {e}. Skipping a sample by assigning it nan score."
            )
            score = np.nan

        return score


@dataclass
class ContextRelevance(MetricWithLLM, SingleTurnMetric):
    """Parameters:
    Score the relevance of the retrieved contexts be based on the user input.

    Input:
        data: list of Dicts with keys: user_input, retrieved_contexts
    Output:
        0.0: retrieved_contexts is not relevant for the user_input
        0.5: retrieved_contexts is partially relevant for the user_input
        1.0: retrieved_contexts is fully relevant for the user_input
    """

    name: str = field(default="nv_context_relevance", repr=True)  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "retrieved_contexts",
            },
        }
    )
    template_relevance1 = (
        "### Instructions\n\n"
        "You are a world class expert designed to evaluate the relevance score of a Context"
        " in order to answer the Question.\n"
        "Your task is to determine if the Context contains proper information to answer the Question.\n"
        "Do not rely on your previous knowledge about the Question.\n"
        "Use only what is written in the Context and in the Question.\n"
        "Follow the instructions below:\n"
        "0. If the context does not contains any relevant information to answer the question, say 0.\n"
        "1. If the context partially contains relevant information to answer the question, say 1.\n"
        "2. If the context contains any relevant information to answer the question, say 2.\n"
        "You must provide the relevance score of 0, 1, or 2, nothing else.\nDo not explain.\n"
        "### Question: {query}\n\n"
        "### Context: {context}\n\n"
        "Do not try to explain.\n"
        "Analyzing Context and Question, the Relevance score is "
    )
    template_relevance2 = (
        "As a specially designed expert to assess the relevance score of a given Context in relation to a Question, "
        "my task is to determine the extent to which the Context provides information necessary to answer the Question. "
        "I will rely solely on the information provided in the Context and Question, and not on any prior knowledge.\n\n"
        "Here are the instructions I will follow:\n"
        "* If the Context does not contain any relevant information to answer the Question, I will respond with a relevance score of 0.\n"
        "* If the Context partially contains relevant information to answer the Question, I will respond with a relevance score of 1.\n"
        "* If the Context contains any relevant information to answer the Question, I will respond with a relevance score of 2.\n\n"
        "### Question: {query}\n\n"
        "### Context: {context}\n\n"
        "Do not try to explain.\n"
        "Based on the provided Question and Context, the Relevance score is  ["
    )
    retry = 5  # Number of retries if rating is not in the first 8 tokens.

    def process_score(self, response):
        for i in [2, 1, 0]:
            if str(i) in response:
                return i / 2
        return np.nan

    def average_scores(self, score0, score1):
        score = np.nan
        if score0 >= 0 and score1 >= 0:
            score = (score0 + score1) / 2
        elif score0 >= 0:
            score = score0
        elif score1 >= 0:
            score = score1
        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not set"
        assert sample.user_input is not None, "User input is not set"
        assert sample.retrieved_contexts is not None, "Retrieved Context is not set"

        if (sample.user_input.strip() == "") or (
            "\n".join(sample.retrieved_contexts).strip() == ""
        ):
            return 0.0
        if sample.user_input.strip() == "\n".join(sample.retrieved_contexts).strip():
            return 0.0
        if "\n".join(sample.retrieved_contexts).strip() in sample.user_input.strip():
            return 0.0

        try:
            score0 = score1 = np.nan
            for retry in range(self.retry):
                formatted_prompt = StringPromptValue(
                    text=self.template_relevance1.format(
                        query=sample.user_input,
                        context="\n".join(sample.retrieved_contexts),
                    )
                )
                req = t.cast(BaseRagasLLM, self.llm).agenerate_text(
                    formatted_prompt,
                    n=1,
                    temperature=0.1,
                )
                resp = await req
                score0 = self.process_score(resp.generations[0][0].text)
                if score0 == score0:
                    break
                else:
                    logger.warning(f"Retry: {retry}")

            for retry in range(self.retry):
                formatted_prompt = StringPromptValue(
                    text=self.template_relevance2.format(
                        query=sample.user_input,
                        context="\n".join(sample.retrieved_contexts),
                    )
                )
                req = t.cast(BaseRagasLLM, self.llm).agenerate_text(
                    formatted_prompt,
                    n=1,
                    temperature=0.1,
                )
                resp = await req
                score1 = self.process_score(resp.generations[0][0].text)
                if score1 == score1:
                    break
                else:
                    logger.warning(f"Retry: {retry}")

            score = self.average_scores(score0, score1)

        except Exception as e:
            print(
                f"An error occurred: {e}. Skipping a sample by assigning it nan score."
            )
            score = np.nan

        return score


@dataclass
class ResponseGroundedness(MetricWithLLM, SingleTurnMetric):
    """Parameters:
    Score the groundedness of the response based on the retrieved contexts.

    Input:
        data: list of Dicts with keys: response, retrieved contexts
    Output:
        0.0: response is not grounded in the retrieved contexts
        0.5: response is partially grounded in the retrieved contexts
        1.0: response is fully grounded in the retrieved contexts
    """

    name: str = field(default="nv_response_groundedness", repr=True)  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "response",
                "retrieved_contexts",
            },
        }
    )
    template_groundedness1 = (
        "### Instruction\n\n"
        "You are a world class expert designed to evaluate the groundedness of an assertion.\n"
        "You will be provided with an assertion and a context.\n"
        "Your task is to determine if the assertion is supported by the context.\n"
        "Follow the instructions below:\n"
        "A. If there is no context or no assertion or context is empty or assertion is empty, say 0.\n"
        "B. If the assertion is not supported by the context, say 0.\n"
        "C. If the assertion is partially supported by the context, say 1.\n"
        "D. If the assertion is fully supported by the context, say 2.\n"
        "You must provide a rating of 0, 1, or 2, nothing else.\n\n"
        "### Context:\n"
        "<{context}>\n\n"
        "### Assertion:\n"
        "<{response}>\n\n"
        "Analyzing Context and Response, the Groundedness score is "
    )
    template_groundedness2 = (
        "As a specialist in assessing the strength of connections between statements and their given contexts, "
        "I will evaluate the level of support an assertion receives from the provided context. Follow these guidelines:\n\n"
        "* If the assertion is not supported or context is empty or assertion is empty, assign a score of 0.\n"
        "* If the assertion is partially supported, assign a score of 1.\n"
        "* If the assertion is fully supported, assign a score of 2.\n\n"
        "I will provide a rating of 0, 1, or 2, without any additional information.\n\n"
        "---\n**Context:**\n[{context}]\n\n"
        "**Assertion:**\n[{response}]\n\n"
        "Do not explain."
        "Based on the provided context and response, the Groundedness score is:"
    )
    retry = 5  # Number of retries if rating is not in the first 8 tokens.

    def process_score(self, response):
        for i in [2, 1, 0]:
            if str(i) in response:
                return i / 2
        return np.nan

    def average_scores(self, score0, score1):
        score = np.nan
        if score0 >= 0 and score1 >= 0:
            score = (score0 + score1) / 2
        elif score0 >= 0:
            score = score0
        elif score1 >= 0:
            score = score1
        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not set"
        assert sample.response is not None, "Response is not set"
        assert sample.retrieved_contexts is not None, "Retrieved Context is not set"

        if (sample.response.strip() == "") or (
            "\n".join(sample.retrieved_contexts).strip().strip() == ""
        ):
            return 0.0
        if sample.response.strip() == "\n".join(sample.retrieved_contexts).strip():
            return 1.0
        if sample.response.strip() in "\n".join(sample.retrieved_contexts).strip():
            return 1.0

        try:
            score0 = score1 = np.nan
            for retry in range(self.retry):
                formatted_prompt = StringPromptValue(
                    text=self.template_groundedness1.format(
                        context="\n".join(sample.retrieved_contexts),
                        response=sample.response,
                    )
                )
                req = t.cast(BaseRagasLLM, self.llm).agenerate_text(
                    formatted_prompt,
                    n=1,
                    temperature=0.1,
                )
                resp = await req
                score0 = self.process_score(resp.generations[0][0].text)
                if score0 == score0:
                    break
                else:
                    logger.warning(f"Retry: {retry}")

            for retry in range(self.retry):
                formatted_prompt = StringPromptValue(
                    text=self.template_groundedness2.format(
                        context="\n".join(sample.retrieved_contexts),
                        response=sample.response,
                    )
                )
                req = t.cast(BaseRagasLLM, self.llm).agenerate_text(
                    formatted_prompt,
                    n=1,
                    temperature=0.1,
                )
                resp = await req
                score1 = self.process_score(resp.generations[0][0].text)
                if score1 == score1:
                    break
                else:
                    logger.warning(f"Retry: {retry}")

            score = self.average_scores(score0, score1)

        except Exception as e:
            print(
                f"An error occurred: {e}. Skipping a sample by assigning it nan score."
            )
            score = np.nan

        return score
