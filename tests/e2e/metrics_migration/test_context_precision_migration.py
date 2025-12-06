"""E2E tests for Context Precision metrics migration from v1 to v2."""

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._context_precision import (
    LLMContextPrecisionWithoutReference as LegacyContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference as LegacyContextPrecisionWithReference,
)
from ragas.metrics.collections import (
    ContextPrecision,
    ContextPrecisionWithoutReference,
    ContextPrecisionWithReference,
    ContextUtilization,
)


class TestContextPrecisionE2EMigration:
    """E2E test compatibility between legacy and V2 Context Precision metrics with modern components."""

    @pytest.fixture
    def sample_data(self):
        """Real-world test cases for context precision evaluation."""
        return [
            {
                "user_input": "What is the capital of France?",
                "response": "Paris is the capital of France.",
                "reference": "The capital of France is Paris.",
                "retrieved_contexts": [
                    "Paris is the capital and largest city of France, with a population of over 2 million people.",
                    "Berlin is the capital of Germany and has a rich historical background.",
                ],
                "description": "Mixed relevant/irrelevant contexts - should penalize irrelevant",
            },
            {
                "user_input": "Who developed the theory of relativity?",
                "response": "Albert Einstein developed the theory of relativity.",
                "reference": "Einstein developed the theory of relativity in the early 1900s.",
                "retrieved_contexts": [
                    "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
                    "Einstein published his special theory of relativity in 1905 and general relativity in 1915.",
                    "Isaac Newton developed the laws of motion and universal gravitation.",
                ],
                "description": "Two relevant, one irrelevant - partial precision",
            },
            {
                "user_input": "What is photosynthesis?",
                "response": "Photosynthesis is the process by which plants make energy from sunlight.",
                "reference": "Photosynthesis is how plants convert sunlight into energy using chlorophyll.",
                "retrieved_contexts": [
                    "Photosynthesis is the process by which plants use sunlight, carbon dioxide, and water to produce glucose.",
                    "During photosynthesis, chlorophyll in plant leaves absorbs light energy to drive the reaction.",
                    "Plants also undergo cellular respiration to break down glucose for energy.",
                ],
                "description": "All contexts relevant to photosynthesis - should score high",
            },
        ]

    @pytest.fixture
    def test_llm(self):
        """Create a LangChain LLM for legacy context precision evaluation."""
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
    async def test_legacy_vs_v2_context_precision_with_reference_e2e_compatibility(
        self, sample_data, test_llm, test_modern_llm
    ):
        """E2E test that legacy and v2 ContextPrecisionWithReference produce similar scores."""

        if test_llm is None or test_modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        for i, data in enumerate(sample_data):
            print(
                f"\n🧪 Testing ContextPrecisionWithReference - Case {i + 1}: {data['description']}"
            )
            print(f"   Question: {data['user_input']}")
            print(f"   Reference: {data['reference'][:60]}...")
            print(f"   Contexts: {len(data['retrieved_contexts'])} context(s)")

            # Legacy implementation
            legacy_metric = LegacyContextPrecisionWithReference(llm=test_llm)
            legacy_sample = SingleTurnSample(
                user_input=data["user_input"],
                reference=data["reference"],
                retrieved_contexts=data["retrieved_contexts"],
            )
            legacy_score = await legacy_metric._single_turn_ascore(legacy_sample, None)

            # V2 implementation
            v2_metric = ContextPrecisionWithReference(llm=test_modern_llm)
            v2_result = await v2_metric.ascore(
                user_input=data["user_input"],
                reference=data["reference"],
                retrieved_contexts=data["retrieved_contexts"],
            )

            score_diff = abs(legacy_score - v2_result.value)
            print(f"   Legacy: {legacy_score:.6f}")
            print(f"   V2:     {v2_result.value:.6f}")
            print(f"   Diff:   {score_diff:.6f}")

            # Context precision should be highly consistent with identical prompts
            assert score_diff < 0.05, (
                f"Legacy and V2 scores should be very similar: Legacy={legacy_score:.6f}, "
                f"V2={v2_result.value:.6f}, Diff={score_diff:.6f} (tolerance: 0.05)"
            )
            print("   ✅ Both implementations give consistent scores")

            # Validate score ranges
            assert 0.0 <= legacy_score <= 1.0
            assert 0.0 <= v2_result.value <= 1.0

    @pytest.mark.asyncio
    async def test_legacy_vs_v2_context_precision_without_reference_e2e_compatibility(
        self, sample_data, test_llm, test_modern_llm
    ):
        """E2E test that legacy and v2 ContextPrecisionWithoutReference produce similar scores."""

        if test_llm is None or test_modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        for i, data in enumerate(sample_data):
            print(
                f"\n🧪 Testing ContextPrecisionWithoutReference - Case {i + 1}: {data['description']}"
            )
            print(f"   Question: {data['user_input']}")
            print(f"   Response: {data['response'][:60]}...")
            print(f"   Contexts: {len(data['retrieved_contexts'])} context(s)")

            # Legacy implementation
            legacy_metric = LegacyContextPrecisionWithoutReference(llm=test_llm)
            legacy_sample = SingleTurnSample(
                user_input=data["user_input"],
                response=data["response"],
                retrieved_contexts=data["retrieved_contexts"],
            )
            legacy_score = await legacy_metric._single_turn_ascore(legacy_sample, None)

            # V2 implementation
            v2_metric = ContextPrecisionWithoutReference(llm=test_modern_llm)
            v2_result = await v2_metric.ascore(
                user_input=data["user_input"],
                response=data["response"],
                retrieved_contexts=data["retrieved_contexts"],
            )

            score_diff = abs(legacy_score - v2_result.value)
            print(f"   Legacy: {legacy_score:.6f}")
            print(f"   V2:     {v2_result.value:.6f}")
            print(f"   Diff:   {score_diff:.6f}")

            # Context precision should be highly consistent with identical prompts
            assert score_diff < 0.05, (
                f"Legacy and V2 scores should be very similar: Legacy={legacy_score:.6f}, "
                f"V2={v2_result.value:.6f}, Diff={score_diff:.6f} (tolerance: 0.05)"
            )
            print("   ✅ Both implementations give consistent scores")

            # Validate score ranges
            assert 0.0 <= legacy_score <= 1.0
            assert 0.0 <= v2_result.value <= 1.0

    @pytest.mark.asyncio
    async def test_context_precision_input_validation(self, test_modern_llm):
        """Test that v2 implementations validate inputs correctly."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for validation testing")

        # Test ContextPrecisionWithReference
        with_ref_metric = ContextPrecisionWithReference(llm=test_modern_llm)

        # Test empty user_input
        with pytest.raises(ValueError, match="user_input cannot be empty"):
            await with_ref_metric.ascore(
                user_input="", reference="valid", retrieved_contexts=["valid"]
            )

        # Test empty reference
        with pytest.raises(ValueError, match="reference cannot be empty"):
            await with_ref_metric.ascore(
                user_input="valid", reference="", retrieved_contexts=["valid"]
            )

        # Test empty retrieved_contexts
        with pytest.raises(ValueError, match="retrieved_contexts cannot be empty"):
            await with_ref_metric.ascore(
                user_input="valid", reference="valid", retrieved_contexts=[]
            )

        # Test ContextPrecisionWithoutReference
        without_ref_metric = ContextPrecisionWithoutReference(llm=test_modern_llm)

        # Test empty response
        with pytest.raises(ValueError, match="response cannot be empty"):
            await without_ref_metric.ascore(
                user_input="valid", response="", retrieved_contexts=["valid"]
            )

    def test_context_precision_migration_requirements_documented(self):
        """Test that migration requirements are properly documented."""

        # V2 implementations should not accept legacy components
        with pytest.raises((TypeError, ValueError, AttributeError)):
            ContextPrecisionWithReference(llm="invalid_llm_type")

        with pytest.raises((TypeError, ValueError, AttributeError)):
            ContextPrecisionWithoutReference(llm=None)

    @pytest.mark.asyncio
    async def test_context_precision_edge_cases(self, test_modern_llm):
        """Test edge cases for context precision metrics."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for edge case testing")

        # Test with single context (should work fine)
        with_ref_metric = ContextPrecisionWithReference(llm=test_modern_llm)
        result = await with_ref_metric.ascore(
            user_input="What is 2+2?",
            reference="2+2 equals 4",
            retrieved_contexts=["In mathematics, 2+2 equals 4."],
        )
        assert 0.0 <= result.value <= 1.0

        # Test without reference variant
        without_ref_metric = ContextPrecisionWithoutReference(llm=test_modern_llm)
        result = await without_ref_metric.ascore(
            user_input="What is 2+2?",
            response="2+2 equals 4",
            retrieved_contexts=["In mathematics, 2+2 equals 4."],
        )
        assert 0.0 <= result.value <= 1.0

    @pytest.mark.asyncio
    async def test_context_precision_wrappers(self, test_modern_llm):
        """Test that the wrapper classes work identically to their base classes."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for wrapper testing")

        test_data = {
            "user_input": "What is the capital of France?",
            "reference": "Paris is the capital of France.",
            "response": "Paris is the capital of France.",
            "retrieved_contexts": ["Paris is the capital and largest city of France."],
        }

        # Test ContextPrecision wrapper vs ContextPrecisionWithReference
        wrapper = ContextPrecision(llm=test_modern_llm)
        base = ContextPrecisionWithReference(llm=test_modern_llm)

        wrapper_result = await wrapper.ascore(
            user_input=test_data["user_input"],
            reference=test_data["reference"],
            retrieved_contexts=test_data["retrieved_contexts"],
        )

        base_result = await base.ascore(
            user_input=test_data["user_input"],
            reference=test_data["reference"],
            retrieved_contexts=test_data["retrieved_contexts"],
        )

        # Should have the correct names
        assert wrapper.name == "context_precision"
        assert base.name == "context_precision_with_reference"

        # Should produce identical scores
        assert wrapper_result.value == base_result.value
        print(f"✅ ContextPrecision wrapper works correctly: {wrapper_result.value}")

        # Test ContextUtilization wrapper vs ContextPrecisionWithoutReference
        wrapper2 = ContextUtilization(llm=test_modern_llm)
        base2 = ContextPrecisionWithoutReference(llm=test_modern_llm)

        wrapper2_result = await wrapper2.ascore(
            user_input=test_data["user_input"],
            response=test_data["response"],
            retrieved_contexts=test_data["retrieved_contexts"],
        )

        base2_result = await base2.ascore(
            user_input=test_data["user_input"],
            response=test_data["response"],
            retrieved_contexts=test_data["retrieved_contexts"],
        )

        # Should have the correct names
        assert wrapper2.name == "context_utilization"
        assert base2.name == "context_precision_without_reference"

        # Should produce identical scores
        assert wrapper2_result.value == base2_result.value
        print(f"✅ ContextUtilization wrapper works correctly: {wrapper2_result.value}")
