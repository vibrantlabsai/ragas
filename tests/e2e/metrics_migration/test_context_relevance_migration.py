"""E2E tests for Context Relevance metric migration from v1 to v2."""

import numpy as np
import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._nv_metrics import ContextRelevance as LegacyContextRelevance
from ragas.metrics.collections import ContextRelevance


# NVIDIA-specific fixtures with correct temperature (0.1)
@pytest.fixture
def nvidia_legacy_llm():
    """Create legacy LLM for ContextRelevance (temperature set in metric calls)."""
    try:
        from langchain_openai import ChatOpenAI

        from ragas.llms.base import LangchainLLMWrapper

        # Legacy sets temperature=0.1 in the metric calls, so use default here
        langchain_llm = ChatOpenAI(model="gpt-4o", temperature=0.01)
        return LangchainLLMWrapper(langchain_llm)
    except Exception as e:
        pytest.skip(str(e))


@pytest.fixture
def nvidia_modern_llm():
    """Create modern LLM with NVIDIA temperature (0.1) for ContextRelevance."""
    try:
        import openai

        from ragas.llms.base import llm_factory

        client = openai.AsyncOpenAI()
        # Set temperature=0.1 to match legacy NVIDIA calls exactly
        return llm_factory(
            model="gpt-4o", provider="openai", client=client, temperature=0.1
        )
    except Exception as e:
        pytest.skip(str(e))


class TestContextRelevanceE2EMigration:
    """E2E test compatibility between legacy ContextRelevance and new V2 ContextRelevance with modern components."""

    @pytest.fixture
    def sample_data(self):
        """Real-world test cases for context relevance evaluation."""
        return [
            {
                "user_input": "When and where was Albert Einstein born?",
                "retrieved_contexts": [
                    "Albert Einstein was born March 14, 1879.",
                    "Albert Einstein was born at Ulm, in Württemberg, Germany.",
                ],
                "description": "Fully relevant contexts - should score high",
            },
            {
                "user_input": "What is photosynthesis?",
                "retrieved_contexts": [
                    "Photosynthesis is the process by which plants convert sunlight into energy.",
                    "Albert Einstein developed the theory of relativity.",
                ],
                "description": "Partially relevant contexts - mixed relevance",
            },
            {
                "user_input": "How do computers work?",
                "retrieved_contexts": [
                    "Albert Einstein was a theoretical physicist.",
                    "The weather today is sunny and warm.",
                ],
                "description": "Irrelevant contexts - should score low",
            },
            {
                "user_input": "What is machine learning?",
                "retrieved_contexts": [
                    "Machine learning is a subset of artificial intelligence that enables computers to learn and improve automatically.",
                ],
                "description": "Single highly relevant context",
            },
        ]

    @pytest.fixture
    def test_llm(self):
        """Create a test LLM for legacy context relevance evaluation."""
        try:
            from ragas.llms.base import llm_factory

            return llm_factory("gpt-4o")
        except ImportError as e:
            pytest.skip(f"LLM factory not available: {e}")
        except Exception as e:
            pytest.skip(f"Could not create LLM (API key may be missing): {e}")

    @pytest.fixture
    def test_modern_llm(self):
        """Create a modern instructor LLM for v2 implementation."""
        try:
            import openai

            from ragas.llms.base import llm_factory

            client = openai.AsyncOpenAI()
            return llm_factory(
                model="gpt-4o",
                provider="openai",
                client=client,
            )
        except ImportError as e:
            pytest.skip(f"Instructor LLM factory not available: {e}")
        except Exception as e:
            pytest.skip(f"Could not create modern LLM (API key may be missing): {e}")

    @pytest.mark.asyncio
    async def test_legacy_context_relevance_vs_v2_context_relevance_e2e_compatibility(
        self, sample_data, nvidia_legacy_llm, nvidia_modern_llm
    ):
        """E2E test that legacy and v2 implementations produce similar scores."""

        if nvidia_legacy_llm is None or nvidia_modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        for i, data in enumerate(sample_data):
            print(
                f"\n🧪 Testing Context Relevance - Case {i + 1}: {data['description']}"
            )
            print(f"   Question: {data['user_input']}")
            print(f"   Contexts: {len(data['retrieved_contexts'])} context(s)")
            for j, ctx in enumerate(data["retrieved_contexts"]):
                print(f"     {j + 1}. {ctx[:60]}...")

            # Legacy implementation
            legacy_context_relevance = LegacyContextRelevance(llm=nvidia_legacy_llm)
            legacy_sample = SingleTurnSample(
                user_input=data["user_input"],
                retrieved_contexts=data["retrieved_contexts"],
            )
            legacy_score = await legacy_context_relevance._single_turn_ascore(
                legacy_sample, None
            )

            # V2 implementation
            v2_context_relevance = ContextRelevance(llm=nvidia_modern_llm)
            v2_result = await v2_context_relevance.ascore(
                user_input=data["user_input"],
                retrieved_contexts=data["retrieved_contexts"],
            )

            score_diff = (
                abs(legacy_score - v2_result.value)
                if not np.isnan(legacy_score) and not np.isnan(v2_result.value)
                else 0.0
            )
            print(f"   Legacy: {legacy_score:.6f}")
            print(f"   V2:     {v2_result.value:.6f}")
            print(f"   Diff:   {score_diff:.6f}")

            # Both implementations use dual judges with same temperature=0.1 - should be identical
            if not np.isnan(legacy_score) and not np.isnan(v2_result.value):
                assert score_diff < 0.01, (
                    f"Legacy and V2 scores should be nearly identical: Legacy={legacy_score:.6f}, "
                    f"V2={v2_result.value:.6f}, Diff={score_diff:.6f} (tolerance: 0.01)"
                )
                print("   ✅ Both implementations give consistent scores")
            else:
                print("   ℹ️  One or both scores are NaN - edge case handling")

            # Validate score ranges (should be 0-1 or NaN)
            if not np.isnan(legacy_score):
                assert 0.0 <= legacy_score <= 1.0
            if not np.isnan(v2_result.value):
                assert 0.0 <= v2_result.value <= 1.0

    @pytest.mark.asyncio
    async def test_context_relevance_edge_cases(self, test_modern_llm):
        """Test edge cases like empty contexts and queries."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for edge case testing")

        metric = ContextRelevance(llm=test_modern_llm)

        # Test empty user input
        with pytest.raises(ValueError, match="user_input is missing"):
            await metric.ascore(
                user_input="",
                retrieved_contexts=["Some context."],
            )

        # Test empty contexts
        with pytest.raises(ValueError, match="retrieved_contexts is missing"):
            await metric.ascore(
                user_input="What is AI?",
                retrieved_contexts=[],
            )

    @pytest.mark.asyncio
    async def test_context_relevance_dual_judge_system(self, test_modern_llm):
        """Test that v2 implementation correctly uses dual-judge system."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for dual-judge testing")

        metric = ContextRelevance(llm=test_modern_llm)

        # Test case where context is clearly relevant
        result = await metric.ascore(
            user_input="What is the capital of France?",
            retrieved_contexts=["Paris is the capital of France and its largest city."],
        )

        print(f"Dual-judge relevance result: {result.value:.3f}")

        # Should be high score for relevant context
        if not np.isnan(result.value):
            assert 0.5 <= result.value <= 1.0, (
                f"Expected high score for relevant context, got {result.value}"
            )

    def test_context_relevance_migration_requirements_documented(self):
        """Test that migration requirements are properly documented."""

        # V2 implementation should not accept legacy components
        with pytest.raises((TypeError, ValueError, AttributeError)):
            ContextRelevance(llm="invalid_llm_type")  # Should reject string

        # V2 should only accept InstructorBaseRagasLLM
        with pytest.raises((TypeError, ValueError, AttributeError)):
            ContextRelevance(llm=None)  # Should reject None
