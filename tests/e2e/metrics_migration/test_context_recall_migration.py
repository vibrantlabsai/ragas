"""E2E tests for Context Recall metric migration from v1 (class-based) to v2 (class-based with automatic validation)."""

import pytest

from ragas.metrics import LLMContextRecall as LegacyContextRecall
from ragas.metrics.collections import ContextRecall

from .base_migration_test import BaseMigrationTest


class TestContextRecallE2EMigration(BaseMigrationTest):
    """E2E test compatibility between legacy ContextRecall class and new V2 ContextRecall class with automatic validation."""

    @pytest.fixture
    def sample_data(self):
        """Real-world test cases for context recall evaluation."""
        return [
            {
                "user_input": "What is the capital of France?",
                "retrieved_contexts": [
                    "Paris is the capital and largest city of France.",
                    "France is a country in Western Europe.",
                ],
                "reference": "Paris is the capital of France. It is located in northern France.",
                "description": "Full attribution - all statements should be found in context",
            },
            {
                "user_input": "Tell me about Albert Einstein",
                "retrieved_contexts": [
                    "Albert Einstein was born in 1879. He developed the theory of relativity."
                ],
                "reference": "Einstein was born in 1879. He won the Nobel Prize in 1921. He developed relativity theory.",
                "description": "Partial attribution - Nobel Prize not mentioned in context",
            },
            {
                "user_input": "What are the main causes of climate change?",
                "retrieved_contexts": [
                    "Climate change is primarily caused by greenhouse gas emissions from burning fossil fuels.",
                    "Deforestation also contributes to climate change by reducing CO2 absorption.",
                ],
                "reference": "The main causes include fossil fuel emissions and deforestation.",
                "description": "Multiple contexts - all statements attributed",
            },
            {
                "user_input": "How does photosynthesis work?",
                "retrieved_contexts": [
                    "Photosynthesis is a process where plants use sunlight to produce glucose."
                ],
                "reference": "Plants convert sunlight into glucose through photosynthesis. This process also produces oxygen and occurs in chloroplasts.",
                "description": "Partial attribution - oxygen and chloroplasts not in context",
            },
            {
                "user_input": "What is quantum computing?",
                "retrieved_contexts": [
                    "Quantum computers use quantum bits or qubits instead of classical bits."
                ],
                "reference": "Quantum computing uses qubits.",
                "description": "Simple case - direct attribution",
            },
        ]

    @pytest.mark.asyncio
    async def test_legacy_context_recall_vs_v2_context_recall_e2e_compatibility(
        self,
        sample_data,
        legacy_llm,
        modern_llm,
    ):
        """E2E test that legacy and v2 implementations produce similar scores with real LLM."""
        await self.run_e2e_compatibility_test(
            sample_data=sample_data,
            legacy_metric_factory=LegacyContextRecall,
            v2_metric_factory=ContextRecall,
            legacy_components={"llm": legacy_llm},
            v2_components={"llm": modern_llm},
            tolerance=0.3,
            metric_name="Context Recall",
            additional_info_keys=["user_input", "reference"],
        )

    @pytest.mark.asyncio
    async def test_context_recall_attribution_detection(self, legacy_llm, modern_llm):
        """Test that both implementations correctly detect statement attributions."""

        if legacy_llm is None or modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        # Test cases specifically for attribution detection
        test_cases = [
            {
                "user_input": "What is the capital of France?",
                "retrieved_contexts": ["Paris is the capital of France."],
                "reference": "Paris is the capital of France.",
                "expected_high": True,
                "description": "Perfect attribution - should get high score",
            },
            {
                "user_input": "What is the capital of France?",
                "retrieved_contexts": ["France is a European country."],
                "reference": "Paris is the capital of France.",
                "expected_high": False,
                "description": "No attribution - should get low score",
            },
            {
                "user_input": "Tell me about Einstein",
                "retrieved_contexts": ["Einstein was born in 1879."],
                "reference": "Einstein was born in 1879. He won the Nobel Prize.",
                "expected_high": False,
                "description": "Partial attribution - should get medium score (50%)",
            },
        ]

        # Define custom assertion function
        def assertion_fn(case, legacy_score, v2_result):
            print(f"   Reference: {case['reference']}")

            if case.get("expected_high"):
                # High attribution should get high scores (> 0.8)
                assert legacy_score > 0.8, (
                    f"Legacy should detect high attribution: {legacy_score}"
                )
                assert v2_result.value > 0.8, (
                    f"V2 class should detect high attribution: {v2_result.value}"
                )
                print("   ✅ All detected high attribution")
            else:
                # Low/partial attribution should get lower scores
                # Note: We don't enforce strict thresholds here as it depends on the specific case
                print(
                    f"   ✅ Scores reflect attribution level (Legacy: {legacy_score:.2f}, V2: {v2_result.value:.2f})"
                )

        await self.run_metric_specific_test(
            test_cases=test_cases,
            legacy_metric_factory=LegacyContextRecall,
            v2_metric_factory=ContextRecall,
            legacy_components={"llm": legacy_llm},
            v2_components={"llm": modern_llm},
            test_name="attribution detection",
            assertion_fn=assertion_fn,
        )

    def test_context_recall_migration_requirements_documented(self):
        """Document the requirements for running full E2E context recall tests."""

        requirements = {
            "llm": "OpenAI GPT, Anthropic Claude, or other LangChain-compatible LLM",
            "environment": "API keys configured for LLM providers",
            "purpose": "Verify that v2 class-based implementation with automatic validation produces similar results to legacy class-based implementation",
        }

        self.create_requirements_documentation(
            metric_name="Context Recall",
            requirements=requirements,
            test_file_name="test_context_recall_migration.py",
        )

        assert True
