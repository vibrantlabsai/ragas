from __future__ import annotations

import asyncio
import unittest
from typing import Any, Dict, List
from unittest.mock import Mock, patch

from pydantic import BaseModel

from ragas.llms.adapters import ADAPTERS, auto_detect_adapter, get_adapter
from ragas.llms.adapters.typesafe import TypeSafeAdapter
from ragas.llms.base import llm_factory
from ragas.llms.typesafe_llm import TypeSafeStructuredLLM


# Sample Pydantic models matching Ragas metric schemas
class StatementFaithfulnessAnswer(BaseModel):
    statement: str
    reason: str
    verdict: int


class NLIStatementOutput(BaseModel):
    statements: List[StatementFaithfulnessAnswer]


class ContextPrecisionOutput(BaseModel):
    reason: str
    verdict: int


class ContextRecallClassification(BaseModel):
    statement: str
    reason: str
    attributed: int


class ContextRecallOutput(BaseModel):
    classifications: List[ContextRecallClassification]


class RubricScoreOutput(BaseModel):
    score: int
    reason: str


class StatementGeneratorOutput(BaseModel):
    statements: List[str]


class MockNoulAnswer:
    def __init__(self, noul: float):
        self.noul = noul


class MockScoreAnswer:
    def __init__(self, score: int, confidence: float = 1.0):
        self.score = score
        self.confidence = confidence


class MockTypeSafeResponse:
    def __init__(
        self,
        nouls: Dict[str, Any] | None = None,
        scores: Dict[str, Any] | None = None,
        choices: Dict[str, Any] | None = None,
    ):
        self.nouls = nouls or {}
        self.scores = scores or {}
        self.choices = choices or {}


class MockSyncTypeSafeClient:
    def __init__(self, response: MockTypeSafeResponse | None = None):
        self.response = response or MockTypeSafeResponse()
        self.calls = []

    def system_one(
        self, state: Any, questions: Dict[str, Any], model: str = "jev-latest"
    ):
        self.calls.append((state, questions, model))
        return self.response


class MockAsyncTypeSafeClient:
    def __init__(self, response: MockTypeSafeResponse | None = None):
        self.response = response or MockTypeSafeResponse()
        self.calls = []

    async def system_one(
        self, state: Any, questions: Dict[str, Any], model: str = "jev-latest"
    ):
        self.calls.append((state, questions, model))
        return self.response


class TestTypeSafeAdapter(unittest.TestCase):
    def test_adapter_registration(self):
        """Test that TypeSafeAdapter is properly registered in ADAPTERS."""
        self.assertIn("typesafe", ADAPTERS)
        adapter = get_adapter("typesafe")
        self.assertIsInstance(adapter, TypeSafeAdapter)

    def test_auto_detect_adapter(self):
        """Test auto-detection for typesafe and jev providers."""
        client = MockSyncTypeSafeClient()
        self.assertEqual(auto_detect_adapter(client, "typesafe"), "typesafe")
        self.assertEqual(auto_detect_adapter(client, "jev"), "typesafe")

    def test_llm_factory_typesafe(self):
        """Test that llm_factory creates a TypeSafeStructuredLLM instance."""
        client = MockSyncTypeSafeClient()
        llm = llm_factory("jev-latest", provider="typesafe", client=client)
        self.assertIsInstance(llm, TypeSafeStructuredLLM)
        self.assertEqual(llm.model, "jev-latest")
        self.assertEqual(llm.provider, "typesafe")

    def test_nli_statement_sync(self):
        """Test batch NLI statement evaluation for Faithfulness."""
        mock_response = MockTypeSafeResponse(
            nouls={
                "stmt_0": MockNoulAnswer(noul=0.92),
                "stmt_1": MockNoulAnswer(noul=0.18),
            }
        )
        client = MockSyncTypeSafeClient(response=mock_response)
        llm = TypeSafeStructuredLLM(client=client, model="jev-latest")

        prompt = (
            "context: Albert Einstein was born in Germany in 1879.\n"
            "statements: ['Einstein was born in Germany.', 'Einstein was born in France.']"
        )
        result = llm.generate(prompt, NLIStatementOutput)

        self.assertIsInstance(result, NLIStatementOutput)
        self.assertEqual(len(result.statements), 2)
        self.assertEqual(result.statements[0].verdict, 1)
        self.assertEqual(result.statements[1].verdict, 0)
        self.assertIn("0.920", result.statements[0].reason)
        self.assertIn("0.180", result.statements[1].reason)

    def test_nli_statement_async(self):
        """Test async batch NLI statement evaluation."""
        mock_response = MockTypeSafeResponse(
            nouls={
                "stmt_0": MockNoulAnswer(noul=0.85),
            }
        )
        client = MockAsyncTypeSafeClient(response=mock_response)
        llm = TypeSafeStructuredLLM(client=client, model="jev-latest")

        prompt = (
            "context: The Earth orbits the Sun.\n"
            "statements: ['The Earth orbits the Sun.']"
        )

        async def run_test():
            return await llm.agenerate(prompt, NLIStatementOutput)

        result = asyncio.run(run_test())
        self.assertEqual(len(result.statements), 1)
        self.assertEqual(result.statements[0].verdict, 1)

    def test_context_precision_verdict(self):
        """Test binary verdict evaluation for Context Precision."""
        mock_response = MockTypeSafeResponse(
            nouls={"verdict": MockNoulAnswer(noul=0.78)}
        )
        client = MockSyncTypeSafeClient(response=mock_response)
        llm = TypeSafeStructuredLLM(client=client, model="jev-latest")

        prompt = (
            "Question: Where is Paris? Context: Paris is in France. Answer: France."
        )
        result = llm.generate(prompt, ContextPrecisionOutput)

        self.assertIsInstance(result, ContextPrecisionOutput)
        self.assertEqual(result.verdict, 1)
        self.assertIn("0.780", result.reason)

    def test_context_recall_classification(self):
        """Test statement attribution evaluation for Context Recall."""
        mock_response = MockTypeSafeResponse(
            nouls={
                "recall_0": MockNoulAnswer(noul=0.89),
                "recall_1": MockNoulAnswer(noul=0.12),
            }
        )
        client = MockSyncTypeSafeClient(response=mock_response)
        llm = TypeSafeStructuredLLM(client=client, model="jev-latest")

        prompt = (
            "context: Quantum mechanics deals with microscopic systems.\n"
            "statements: ['Quantum mechanics studies microscopic scales.', 'Quantum mechanics explains dinosaurs.']"
        )
        result = llm.generate(prompt, ContextRecallOutput)

        self.assertIsInstance(result, ContextRecallOutput)
        self.assertEqual(len(result.classifications), 2)
        self.assertEqual(result.classifications[0].attributed, 1)
        self.assertEqual(result.classifications[1].attributed, 0)

    def test_rubric_scoring(self):
        """Test ordinal rubric score evaluation."""
        mock_response = MockTypeSafeResponse(
            scores={"rubric": MockScoreAnswer(score=5, confidence=0.96)}
        )
        client = MockSyncTypeSafeClient(response=mock_response)
        llm = TypeSafeStructuredLLM(client=client, model="jev-latest")

        prompt = (
            "Evaluate the accuracy and comprehensiveness of this technical response."
        )
        result = llm.generate(prompt, RubricScoreOutput)

        self.assertIsInstance(result, RubricScoreOutput)
        self.assertEqual(result.score, 5)
        self.assertIn("0.96", result.reason)

    def test_hybrid_open_ended_generation_with_generator(self):
        """Test open-ended text synthesis delegates to generator_llm."""
        generator_mock = Mock()
        expected = StatementGeneratorOutput(
            statements=["Sentence one.", "Sentence two."]
        )
        generator_mock.generate.return_value = expected

        client = MockSyncTypeSafeClient()
        llm = TypeSafeStructuredLLM(
            client=client,
            model="jev-latest",
            generator_llm=generator_mock,
        )

        prompt = "Break down the answer into sentences."
        result = llm.generate(prompt, StatementGeneratorOutput)

        self.assertEqual(result.statements, ["Sentence one.", "Sentence two."])
        generator_mock.generate.assert_called_once_with(
            prompt, StatementGeneratorOutput
        )

    def test_open_ended_generation_without_generator_raises(self):
        """Test open-ended text synthesis raises informative ValueError if generator_llm is missing."""
        client = MockSyncTypeSafeClient()
        llm = TypeSafeStructuredLLM(client=client, model="jev-latest")

        prompt = "Break down the answer into sentences."
        with self.assertRaises(ValueError) as ctx:
            llm.generate(prompt, StatementGeneratorOutput)

        self.assertIn("System One decision model", str(ctx.exception))
        self.assertIn("generator_llm", str(ctx.exception))

    def test_missing_typesafe_sdk_raises_import_error(self):
        """Test that missing typesafe-sdk raises an informative ImportError when client is not passed."""
        llm = TypeSafeStructuredLLM(client=None, api_key="test-key")
        with patch.dict("sys.modules", {"typesafe_sdk": None}):
            with self.assertRaises(ImportError) as ctx:
                llm._get_client(async_mode=False)
            self.assertIn("pip install typesafe-sdk", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
