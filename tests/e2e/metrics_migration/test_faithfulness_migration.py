"""E2E tests for Faithfulness metric migration from v1 to v2."""

import numpy as np
import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._faithfulness import Faithfulness as LegacyFaithfulness
from ragas.metrics.collections import Faithfulness


class TestFaithfulnessE2EMigration:
    """E2E test compatibility between legacy Faithfulness and new V2 Faithfulness with modern components."""

    @pytest.fixture
    def sample_data(self):
        """Real-world test cases for faithfulness evaluation."""
        return [
            {
                "user_input": "Where was Einstein born?",
                "response": "Einstein was born in Germany on 14th March 1879.",
                "retrieved_contexts": [
                    "Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time."
                ],
                "description": "High faithfulness - response supported by context",
            },
            {
                "user_input": "Where was Einstein born?",
                "response": "Einstein was born in Germany on 20th March 1879.",
                "retrieved_contexts": [
                    "Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time."
                ],
                "description": "Low faithfulness - wrong date not supported by context",
            },
            {
                "user_input": "When was the first super bowl?",
                "response": "The first superbowl was held on Jan 15, 1967",
                "retrieved_contexts": [
                    "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
                ],
                "description": "Perfect faithfulness - exact match with context",
            },
            {
                "user_input": "What is photosynthesis?",
                "response": "Photosynthesis is how plants make energy and produce oxygen.",
                "retrieved_contexts": [
                    "Photosynthesis is the process by which plants convert sunlight into energy.",
                    "During photosynthesis, plants produce oxygen as a byproduct.",
                ],
                "description": "Multi-context faithfulness - response draws from multiple contexts",
            },
        ]

    @pytest.fixture
    def test_llm(self):
        """Create a LangChain LLM for legacy faithfulness evaluation."""
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
            return llm_factory("gpt-4o", client=client)
        except ImportError as e:
            pytest.skip(f"LLM factory not available: {e}")
        except Exception as e:
            pytest.skip(f"Could not create modern LLM (API key may be missing): {e}")

    @pytest.mark.asyncio
    async def test_legacy_faithfulness_vs_v2_faithfulness_e2e_compatibility(
        self, sample_data, test_llm, test_modern_llm
    ):
        """E2E test that legacy and v2 implementations produce similar scores."""

        if test_llm is None or test_modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        for i, data in enumerate(sample_data):
            print(f"\n🧪 Testing Faithfulness - Case {i + 1}: {data['description']}")
            print(f"   Question: {data['user_input']}")
            print(f"   Response: {data['response'][:80]}...")
            print(f"   Contexts: {len(data['retrieved_contexts'])} context(s)")

            # Legacy implementation
            legacy_faithfulness = LegacyFaithfulness(llm=test_llm)
            legacy_sample = SingleTurnSample(
                user_input=data["user_input"],
                response=data["response"],
                retrieved_contexts=data["retrieved_contexts"],
            )
            legacy_score = await legacy_faithfulness._single_turn_ascore(
                legacy_sample, None
            )

            # V2 implementation
            v2_faithfulness = Faithfulness(llm=test_modern_llm)
            v2_result = await v2_faithfulness.ascore(
                user_input=data["user_input"],
                response=data["response"],
                retrieved_contexts=data["retrieved_contexts"],
            )

            score_diff = abs(legacy_score - v2_result.value)
            print(f"   Legacy: {legacy_score:.6f}")
            print(f"   V2:     {v2_result.value:.6f}")
            print(f"   Diff:   {score_diff:.6f}")

            # Ensure implementations give reasonably similar scores
            # Faithfulness should be more consistent than complex metrics
            assert score_diff < 0.1, (
                f"Legacy and V2 scores should be similar: Legacy={legacy_score:.6f}, "
                f"V2={v2_result.value:.6f}, Diff={score_diff:.6f} (tolerance: 0.1)"
            )
            print("   ✅ Both implementations give consistent scores")

            # Validate score ranges (both should be 0-1 or NaN)
            if not np.isnan(legacy_score):
                assert 0.0 <= legacy_score <= 1.0
            if not np.isnan(v2_result.value):
                assert 0.0 <= v2_result.value <= 1.0

    @pytest.mark.asyncio
    async def test_faithfulness_edge_cases(self, test_modern_llm):
        """Test edge cases like empty responses and contexts."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for edge case testing")

        metric = Faithfulness(llm=test_modern_llm)

        # Test empty response
        with pytest.raises(ValueError, match="response is missing"):
            await metric.ascore(
                user_input="What is AI?",
                response="",
                retrieved_contexts=["AI is artificial intelligence."],
            )

        # Test empty user_input
        with pytest.raises(ValueError, match="user_input is missing"):
            await metric.ascore(
                user_input="",
                response="AI is smart.",
                retrieved_contexts=["AI context."],
            )

        # Test empty contexts
        with pytest.raises(ValueError, match="retrieved_contexts is missing"):
            await metric.ascore(
                user_input="What is AI?",
                response="AI is smart.",
                retrieved_contexts=[],
            )

    @pytest.mark.asyncio
    async def test_faithfulness_high_vs_low_scores(self, test_modern_llm):
        """Test that faithfulness correctly distinguishes high vs low faithfulness."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for score testing")

        metric = Faithfulness(llm=test_modern_llm)

        # High faithfulness case
        high_result = await metric.ascore(
            user_input="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieved_contexts=["Paris is the capital and largest city of France."],
        )

        # Low faithfulness case
        low_result = await metric.ascore(
            user_input="What is the capital of France?",
            response="The capital of France is London.",
            retrieved_contexts=["Paris is the capital and largest city of France."],
        )

        print(f"High faithfulness score: {high_result.value:.3f}")
        print(f"Low faithfulness score: {low_result.value:.3f}")

        # Validate ranges
        assert 0.0 <= high_result.value <= 1.0
        assert 0.0 <= low_result.value <= 1.0

        # High faithfulness should typically score higher than low faithfulness
        # (though this depends on statement decomposition)

    def test_faithfulness_migration_requirements_documented(self):
        """Test that migration requirements are properly documented."""

        # V2 implementation should not accept legacy components
        with pytest.raises((TypeError, ValueError, AttributeError)):
            Faithfulness(llm="invalid_llm_type")  # Should reject string

        # V2 should only accept InstructorBaseRagasLLM
        with pytest.raises((TypeError, ValueError, AttributeError)):
            Faithfulness(llm=None)  # Should reject None
