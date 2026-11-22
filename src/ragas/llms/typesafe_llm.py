from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import threading
import typing as t
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from ragas._analytics import LLMUsageEvent, track
from ragas.cache import CacheInterface, cacher
from ragas.llms.base import InstructorBaseRagasLLM, InstructorTypeVar

logger = logging.getLogger(__name__)

DEFAULT_TYPESAFE_MODEL = "jev-latest"


class TypeSafeStructuredLLM(InstructorBaseRagasLLM):
    """
    LLM wrapper for TypeSafe AI's Jev model.

    Jev is a non-autoregressive "System One" decision model that outputs
    calibrated probabilities in 70–500ms rather than generating token streams.
    It specializes in binary verdicts, ordinal rubrics, and classification
    with zero hallucination and sub-second latency.

    For metrics that combine text extraction with decision judging (such as
    Faithfulness, which first breaks an answer into atomic statements and then
    checks each statement against context), an optional `generator_llm` can be
    supplied. When provided, text synthesis is handled by the generator while
    all decision judging is handled by Jev.

    Args:
        client: Pre-initialized TypeSafeClient or AsyncTypeSafeClient instance.
        model: Model name (default: "jev-latest").
        api_key: TypeSafe AI API key (falls back to TYPESAFE_API_KEY env var).
        generator_llm: Optional generative LLM for open-ended text synthesis.
        cache: Optional cache backend for caching responses.
        **kwargs: Additional options forwarded to the client.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = DEFAULT_TYPESAFE_MODEL,
        api_key: Optional[str] = None,
        generator_llm: Optional[InstructorBaseRagasLLM] = None,
        cache: Optional[CacheInterface] = None,
        **kwargs: Any,
    ):
        self.model = model or DEFAULT_TYPESAFE_MODEL
        self.provider = "typesafe"
        self.generator_llm = generator_llm
        self.cache = cache
        self.kwargs = kwargs

        # Resolve API key
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")

        # Client resolution
        self._client = client
        self.is_async = self._check_client_async(client)

        if self.cache is not None:
            self.generate = cacher(cache_backend=self.cache)(self.generate)  # type: ignore
            self.agenerate = cacher(cache_backend=self.cache)(self.agenerate)  # type: ignore

    def _check_client_async(self, client: Optional[Any]) -> bool:
        if client is None:
            return False
        class_name = client.__class__.__name__
        return "Async" in class_name or hasattr(client, "asystem_one")

    def _get_client(self, async_mode: bool = False) -> Any:
        if self._client is not None:
            return self._client

        try:
            import typesafe_sdk
        except ImportError:
            raise ImportError(
                "The `typesafe-sdk` package is required to use TypeSafe AI (Jev).\n"
                "Install it with: pip install typesafe-sdk"
            )

        if not self.api_key:
            raise ValueError(
                "TypeSafe AI API key is required. Pass `api_key` to TypeSafeStructuredLLM "
                "or set the `TYPESAFE_API_KEY` environment variable."
            )

        client_cls = (
            typesafe_sdk.AsyncTypeSafeClient
            if async_mode
            else typesafe_sdk.TypeSafeClient
        )
        return client_cls(api_key=self.api_key, **self.kwargs)

    def _run_async_in_current_loop(self, coro: t.Coroutine) -> Any:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                result_container: Dict[str, Any] = {"result": None, "exception": None}

                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result_container["result"] = new_loop.run_until_complete(coro)
                    except Exception as e:
                        result_container["exception"] = e
                    finally:
                        new_loop.close()

                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()

                if result_container["exception"]:
                    raise result_container["exception"]
                return result_container["result"]
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
                asyncio.set_event_loop(None)

    ################################################################
    # Generation Interface
    ################################################################

    def generate(
        self, prompt: str, response_model: Type[InstructorTypeVar]
    ) -> InstructorTypeVar:
        """Synchronously evaluate prompt using Jev decision engine."""
        if self.is_async:
            return self._run_async_in_current_loop(
                self.agenerate(prompt, response_model)
            )

        client = self._get_client(async_mode=False)
        result = self._execute_evaluation(
            client, prompt, response_model, is_async=False
        )

        track(
            LLMUsageEvent(
                provider=self.provider,
                model=self.model,
                llm_type="typesafe_structured",
                num_requests=1,
                is_async=False,
            )
        )
        return result

    async def agenerate(
        self, prompt: str, response_model: Type[InstructorTypeVar]
    ) -> InstructorTypeVar:
        """Asynchronously evaluate prompt using Jev decision engine."""
        client = self._get_client(async_mode=True)
        result = await self._execute_evaluation(
            client, prompt, response_model, is_async=True
        )

        track(
            LLMUsageEvent(
                provider=self.provider,
                model=self.model,
                llm_type="typesafe_structured",
                num_requests=1,
                is_async=True,
            )
        )
        return result

    ################################################################
    # Decision Translation Engine
    ################################################################

    def _execute_evaluation(
        self,
        client: Any,
        prompt: str,
        response_model: Type[InstructorTypeVar],
        is_async: bool,
    ) -> Any:
        # Check if schema is an open-ended generative task (e.g. StatementGeneratorOutput)
        if self._is_open_ended_generation(response_model):
            if self.generator_llm is not None:
                if is_async:
                    return self.generator_llm.agenerate(prompt, response_model)
                return self.generator_llm.generate(prompt, response_model)
            else:
                raise ValueError(
                    f"TypeSafe Jev is a System One decision model specialized for verdict, "
                    f"classification, and scoring rubrics. The schema '{response_model.__name__}' "
                    f"requires open-ended text synthesis.\n\n"
                    f"To support hybrid evaluation, provide a `generator_llm`:\n"
                    f"  generator = llm_factory('gpt-4o-mini', client=...)\n"
                    f"  llm = llm_factory('jev-latest', provider='typesafe', generator_llm=generator)"
                )

        if is_async:
            return self._run_async_evaluation(client, prompt, response_model)
        return self._run_sync_evaluation(client, prompt, response_model)

    def _is_open_ended_generation(self, model_cls: Type[BaseModel]) -> bool:
        """Detect if schema is for open-ended text synthesis rather than decision judging."""
        schema_name = model_cls.__name__.lower()
        if "statementgenerator" in schema_name or "extractedkeyphrase" in schema_name:
            return True
        fields = getattr(model_cls, "model_fields", {}) or getattr(
            model_cls, "__fields__", {}
        )
        # If the only field is a list of strings and no verdict/score field exists
        if len(fields) == 1:
            for field_name, field_info in fields.items():
                annotation = getattr(field_info, "annotation", None)
                if annotation in (List[str], list[str]):
                    return True
        return False

    def _run_sync_evaluation(
        self, client: Any, prompt: str, response_model: Type[InstructorTypeVar]
    ) -> Any:
        state, questions, reconstructor = self._prepare_jev_call(prompt, response_model)
        response = client.system_one(state, questions, model=self.model)
        return reconstructor(response)

    async def _run_async_evaluation(
        self, client: Any, prompt: str, response_model: Type[InstructorTypeVar]
    ) -> Any:
        state, questions, reconstructor = self._prepare_jev_call(prompt, response_model)
        if hasattr(client, "system_one") and inspect.iscoroutinefunction(
            client.system_one
        ):
            response = await client.system_one(state, questions, model=self.model)
        elif hasattr(client, "asystem_one"):
            response = await client.asystem_one(state, questions, model=self.model)
        else:
            response = client.system_one(state, questions, model=self.model)
        return reconstructor(response)

    ################################################################
    # Schema-to-Jev Mapping
    ################################################################

    def _prepare_jev_call(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> t.Tuple[str, Dict[str, Any], t.Callable[[Any], Any]]:
        """
        Analyze prompt and response_model to construct Jev questions
        and return a reconstructor function to build the Pydantic instance.
        """
        model_name = response_model.__name__.lower()

        # Case 1: Statement-level NLI (e.g. NLIStatementOutput for Faithfulness)
        if "nlistatement" in model_name or "faithfulness" in model_name:
            return self._build_nli_batch(prompt, response_model)

        # Case 2: Statement-level Context Recall (e.g. ContextRecallOutput)
        if "contextrecall" in model_name:
            return self._build_context_recall_batch(prompt, response_model)

        # Case 3: Binary Context Precision (e.g. ContextPrecisionOutput)
        if "contextprecision" in model_name:
            return self._build_binary_verdict(
                prompt,
                response_model,
                instruction="Was the retrieved context useful in answering the question?",
            )

        # Case 4: Answer Accuracy / Correctness
        if "answeraccuracy" in model_name or "correctness" in model_name:
            return self._build_binary_verdict(
                prompt,
                response_model,
                instruction="Is the answer accurate and correct with respect to the reference?",
            )

        # Case 5: Rubric / Ordinal Scoring (e.g. RubricScoreOutput)
        if "rubric" in model_name:
            return self._build_rubric_scoring(prompt, response_model)

        # Default fallback: inspect schema fields dynamically
        return self._build_generic_verdict(prompt, response_model)

    def _build_nli_batch(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> t.Tuple[str, Dict[str, Any], t.Callable[[Any], Any]]:
        """Build batch Noul questions for statement-level NLI evaluation."""
        context, statements = self._extract_context_and_statements(prompt)
        questions: Dict[str, Any] = {}

        for idx, statement in enumerate(statements):
            key = f"stmt_{idx}"
            questions[key] = {
                "type": "noul",
                "instructions": f"Can the statement '{statement}' be directly inferred from the context?",
            }

        # Fallback if no statements extracted
        if not questions:
            questions["stmt_0"] = {
                "type": "noul",
                "instructions": "Is the statement supported by the context?",
            }
            statements = [prompt]

        def reconstruct(response: Any) -> Any:
            fields = (
                response_model.model_fields
                if hasattr(response_model, "model_fields")
                else response_model.__fields__
            )
            items_field_name = next(iter(fields.keys()))
            item_cls = t.get_args(fields[items_field_name].annotation)[0]

            statement_answers = []
            for idx, stmt in enumerate(statements):
                key = f"stmt_{idx}"
                noul_resp = getattr(response, "nouls", {}).get(key)
                prob = (
                    getattr(noul_resp, "noul", 0.0)
                    if noul_resp is not None
                    else (
                        response.get("nouls", {}).get(key, 0.0)
                        if isinstance(response, dict)
                        else 0.0
                    )
                )
                verdict = 1 if prob >= 0.5 else 0
                reason = f"Jev calibrated P(inferred) = {prob:.3f}"

                # Instantiate the child model (e.g. StatementFaithfulnessAnswer)
                statement_answers.append(
                    item_cls(statement=stmt, reason=reason, verdict=verdict)
                )

            return response_model(**{items_field_name: statement_answers})

        return context, questions, reconstruct

    def _build_context_recall_batch(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> t.Tuple[str, Dict[str, Any], t.Callable[[Any], Any]]:
        """Build batch Noul questions for Context Recall evaluation."""
        context, statements = self._extract_context_and_statements(prompt)
        questions: Dict[str, Any] = {}

        for idx, statement in enumerate(statements):
            key = f"recall_{idx}"
            questions[key] = {
                "type": "noul",
                "instructions": f"Can the statement '{statement}' be attributed to the context?",
            }

        if not questions:
            questions["recall_0"] = {
                "type": "noul",
                "instructions": "Can the statement be attributed to the context?",
            }
            statements = [prompt]

        def reconstruct(response: Any) -> Any:
            fields = (
                response_model.model_fields
                if hasattr(response_model, "model_fields")
                else response_model.__fields__
            )
            items_field_name = next(iter(fields.keys()))
            item_cls = t.get_args(fields[items_field_name].annotation)[0]

            classifications = []
            for idx, stmt in enumerate(statements):
                key = f"recall_{idx}"
                noul_resp = getattr(response, "nouls", {}).get(key)
                prob = (
                    getattr(noul_resp, "noul", 0.0)
                    if noul_resp is not None
                    else (
                        response.get("nouls", {}).get(key, 0.0)
                        if isinstance(response, dict)
                        else 0.0
                    )
                )
                attributed = 1 if prob >= 0.5 else 0
                reason = f"Jev calibrated P(attributed) = {prob:.3f}"

                classifications.append(
                    item_cls(statement=stmt, reason=reason, attributed=attributed)
                )

            return response_model(**{items_field_name: classifications})

        return context, questions, reconstruct

    def _build_binary_verdict(
        self, prompt: str, response_model: Type[BaseModel], instruction: str
    ) -> t.Tuple[str, Dict[str, Any], t.Callable[[Any], Any]]:
        """Build single Noul question for binary verdict schemas."""
        questions = {
            "verdict": {
                "type": "noul",
                "instructions": instruction,
            }
        }

        def reconstruct(response: Any) -> Any:
            noul_resp = getattr(response, "nouls", {}).get("verdict")
            prob = (
                getattr(noul_resp, "noul", 0.0)
                if noul_resp is not None
                else (
                    response.get("nouls", {}).get("verdict", 0.0)
                    if isinstance(response, dict)
                    else 0.0
                )
            )
            verdict = 1 if prob >= 0.5 else 0
            reason = f"Jev calibrated probability = {prob:.3f}"
            return response_model(verdict=verdict, reason=reason)

        return prompt, questions, reconstruct

    def _build_rubric_scoring(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> t.Tuple[str, Dict[str, Any], t.Callable[[Any], Any]]:
        """Build Score question for rubric-based scoring (1–5 scale)."""
        questions = {
            "rubric": {
                "type": "score",
                "instructions": "Evaluate the response quality against the specified rubric.",
                "criteria": [
                    "Completely inaccurate or irrelevant",
                    "Major inaccuracies or missing key details",
                    "Partially accurate with minor omissions",
                    "Accurate and relevant with minor stylistic flaws",
                    "Fully accurate, comprehensive, and well-reasoned",
                ],
            }
        }

        def reconstruct(response: Any) -> Any:
            score_resp = getattr(response, "scores", {}).get("rubric")
            score = (
                getattr(score_resp, "score", 3)
                if score_resp is not None
                else (
                    response.get("scores", {}).get("rubric", {}).get("score", 3)
                    if isinstance(response, dict)
                    else 3
                )
            )
            confidence = (
                getattr(score_resp, "confidence", 1.0)
                if score_resp is not None
                else 1.0
            )
            reason = f"Jev calibrated score (confidence: {confidence:.2f})"
            return response_model(score=score, reason=reason)

        return prompt, questions, reconstruct

    def _build_generic_verdict(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> t.Tuple[str, Dict[str, Any], t.Callable[[Any], Any]]:
        """Generic fallback for other verdict schemas."""
        fields = (
            response_model.model_fields
            if hasattr(response_model, "model_fields")
            else response_model.__fields__
        )
        verdict_key = next(
            (f for f in fields if f in ("verdict", "attributed", "result", "is_valid")),
            "verdict",
        )

        questions = {
            verdict_key: {
                "type": "noul",
                "instructions": "Determine if the evaluation criteria is satisfied.",
            }
        }

        def reconstruct(response: Any) -> Any:
            noul_resp = getattr(response, "nouls", {}).get(verdict_key)
            prob = getattr(noul_resp, "noul", 0.0) if noul_resp is not None else 0.0
            verdict_val = 1 if prob >= 0.5 else 0
            reason_val = f"Jev calibrated P = {prob:.3f}"

            kwargs: Dict[str, Any] = {}
            for field_name in fields:
                if field_name == verdict_key:
                    kwargs[field_name] = verdict_val
                elif field_name == "reason":
                    kwargs[field_name] = reason_val
                else:
                    kwargs[field_name] = None
            return response_model(**kwargs)

        return prompt, questions, reconstruct

    def _extract_context_and_statements(self, prompt: str) -> t.Tuple[str, List[str]]:
        """Helper to extract context string and statements list from structured prompts."""
        context = ""
        statements: List[str] = []

        context_match = re.search(
            r"context:\s*(.*?)(?=\nstatements:|\Z)", prompt, re.DOTALL | re.IGNORECASE
        )
        if context_match:
            context = context_match.group(1).strip()
        else:
            context = prompt

        statements_match = re.search(
            r"statements:\s*\[?(.*?)\]?$", prompt, re.DOTALL | re.IGNORECASE
        )
        if statements_match:
            raw_stmts = statements_match.group(1)
            # Find quoted items or bullet points
            items = re.findall(r"['\"](.*?)['\"]", raw_stmts)
            if items:
                statements = [item.strip() for item in items if item.strip()]
            else:
                statements = [
                    line.strip().lstrip("-*0123456789. ")
                    for line in raw_stmts.split("\n")
                    if line.strip()
                ]

        return context, statements

    def __repr__(self) -> str:
        return (
            f"TypeSafeStructuredLLM(model='{self.model}', provider='{self.provider}')"
        )
