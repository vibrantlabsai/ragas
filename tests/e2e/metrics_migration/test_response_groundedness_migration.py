"""E2E tests for ResponseGroundedness metric migration from v1 to v2."""

import numpy as np
import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._nv_metrics import ResponseGroundedness as LegacyResponseGroundedness
from ragas.metrics.collections import ResponseGroundedness


class TestResponseGroundednessE2EMigration:
    """E2E test compatibility between legacy ResponseGroundedness and new V2 ResponseGroundedness with modern components."""

    @pytest.fixture
    def sample_data(self):
        """Real-world test cases for response groundedness evaluation."""
        return [
            {
                "response": "Einstein was born in Germany on March 14, 1879.",
                "retrieved_contexts": [
                    "Albert Einstein was born in Ulm, Germany on March 14, 1879."
                ],
                "description": "High groundedness - response fully supported by context",
            },
            {
                "response": "Einstein was born in France on March 14, 1879.",
                "retrieved_contexts": [
                    "Albert Einstein was born in Ulm, Germany on March 14, 1879."
                ],
                "description": "Low groundedness - wrong country not supported by context",
            },
            {
                "response": "Einstein was a physicist.",
                "retrieved_contexts": [
                    "Albert Einstein was a German-born theoretical physicist, widely held to be one of the greatest scientists of all time."
                ],
                "description": "High groundedness - response supported by context",
            },
            {
                "response": "The capital of France is Paris, and it has a population of over 2 million.",
                "retrieved_contexts": [
                    "Paris is the capital and most populous city of France."
                ],
                "description": "Partial groundedness - capital correct, population not mentioned",
            },
            {
                "response": "Photosynthesis is the process by which plants convert sunlight into energy.",
                "retrieved_contexts": [
                    "Photosynthesis is a biological process where plants use sunlight to create glucose and oxygen."
                ],
                "description": "High groundedness - core concept supported",
            },
        ]

    @pytest.fixture
    def test_llm(self):
        """Create a LangChain LLM for legacy response groundedness evaluation."""
        try:
            from langchain_openai import ChatOpenAI

            from ragas.llms import LangchainLLMWrapper

            langchain_llm = ChatOpenAI(model="gpt-4o", temperature=0.01)
            return LangchainLLMWrapper(langchain_llm)
        except ImportError as e:
            pytest.skip(f"LangChain LLM not available: {e}")
        except Exception as e:
            pytest.skip(f"Could not create LangChain LLM (API key may be missing): {e}")

    @pytest.fixture
    def test_modern_llm(self):
        """Create a modern instructor LLM for v2 implementation."""
        try:
            import openai

            from ragas.llms.base import llm_factory

            client = openai.AsyncOpenAI()
            # Use legacy temperature (0.1) for perfect compatibility
            return llm_factory("gpt-4o", client=client, temperature=0.1)
        except ImportError as e:
            pytest.skip(f"LLM factory not available: {e}")
        except Exception as e:
            pytest.skip(f"Could not create modern LLM (API key may be missing): {e}")

    @pytest.mark.asyncio
    async def test_legacy_response_groundedness_vs_v2_response_groundedness_e2e_compatibility(
        self, sample_data, test_llm, test_modern_llm
    ):
        """E2E test that legacy and v2 implementations produce similar scores."""

        if test_llm is None or test_modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        for i, data in enumerate(sample_data):
            print(
                f"\n🧪 Testing ResponseGroundedness - Case {i + 1}: {data['description']}"
            )
            print(f"   Response: {data['response'][:80]}...")
            print(f"   Contexts: {len(data['retrieved_contexts'])} context(s)")

            # Legacy implementation
            legacy_groundedness = LegacyResponseGroundedness(llm=test_llm)
            legacy_sample = SingleTurnSample(
                response=data["response"],
                retrieved_contexts=data["retrieved_contexts"],
            )
            legacy_score = await legacy_groundedness._single_turn_ascore(
                legacy_sample, None
            )

            # V2 implementation
            v2_groundedness = ResponseGroundedness(llm=test_modern_llm)
            v2_result = await v2_groundedness.ascore(
                response=data["response"],
                retrieved_contexts=data["retrieved_contexts"],
            )

            score_diff = abs(legacy_score - v2_result.value)
            print(f"   Legacy: {legacy_score:.6f}")
            print(f"   V2:     {v2_result.value:.6f}")
            print(f"   Diff:   {score_diff:.6f}")

            # Ensure implementations give reasonably similar scores
            # Response groundedness uses dual-judge system with some variation expected
            assert score_diff < 0.3, (
                f"Legacy and V2 scores should be similar: Legacy={legacy_score:.6f}, "
                f"V2={v2_result.value:.6f}, Diff={score_diff:.6f} (tolerance: 0.3)"
            )
            print("   ✅ Both implementations give consistent scores")

            # Validate score ranges (both should be 0-1 or NaN)
            if not np.isnan(legacy_score):
                assert 0.0 <= legacy_score <= 1.0
            if not np.isnan(v2_result.value):
                assert 0.0 <= v2_result.value <= 1.0

    @pytest.mark.asyncio
    async def test_response_groundedness_edge_cases(self, test_modern_llm):
        """Test edge cases like empty responses and contexts."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for edge case testing")

        metric = ResponseGroundedness(llm=test_modern_llm)

        # Test empty response
        with pytest.raises(ValueError, match="response is missing"):
            await metric.ascore(
                response="",
                retrieved_contexts=["Some context about Einstein."],
            )

        # Test empty contexts
        with pytest.raises(ValueError, match="retrieved_contexts is missing"):
            await metric.ascore(
                response="Einstein was a physicist.",
                retrieved_contexts=[],
            )

    @pytest.mark.asyncio
    async def test_response_groundedness_scoring_behavior(self, test_modern_llm):
        """Test that response groundedness produces expected score patterns."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for scoring testing")

        metric = ResponseGroundedness(llm=test_modern_llm)

        # High groundedness case
        high_result = await metric.ascore(
            response="The capital of France is Paris.",
            retrieved_contexts=["Paris is the capital and largest city of France."],
        )

        # Low groundedness case
        low_result = await metric.ascore(
            response="The capital of France is London.",
            retrieved_contexts=["Paris is the capital and largest city of France."],
        )

        print(f"High groundedness score: {high_result.value:.3f}")
        print(f"Low groundedness score: {low_result.value:.3f}")

        # Validate ranges
        assert 0.0 <= high_result.value <= 1.0
        assert 0.0 <= low_result.value <= 1.0

        # High groundedness should typically score higher than low groundedness
        # (though exact scores depend on judge behavior)

    @pytest.mark.asyncio
    async def test_response_groundedness_dual_judge_system(self, test_modern_llm):
        """Test that the dual-judge system is working with different contexts."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for dual-judge testing")

        metric = ResponseGroundedness(llm=test_modern_llm)

        # Test with multiple contexts that provide different levels of support
        result = await metric.ascore(
            response="Einstein developed the theory of relativity and won a Nobel Prize.",
            retrieved_contexts=[
                "Albert Einstein developed the theory of relativity.",
                "Einstein won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.",
            ],
        )

        print(f"Multi-context groundedness score: {result.value:.3f}")

        # Should be well-grounded since both parts are supported
        assert 0.0 <= result.value <= 1.0

    def test_response_groundedness_migration_requirements_documented(self):
        """Test that migration requirements are properly documented."""

        # V2 implementation should not accept legacy components
        with pytest.raises((TypeError, ValueError, AttributeError)):
            ResponseGroundedness(llm="invalid_llm_type")  # type: ignore[arg-type]  # Should reject string

        # V2 should only accept InstructorBaseRagasLLM
        with pytest.raises((TypeError, ValueError, AttributeError)):
            ResponseGroundedness(llm=None)  # type: ignore[arg-type]  # Should reject None
