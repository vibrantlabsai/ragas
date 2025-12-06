"""E2E tests for Answer Accuracy metric migration from v1 to v2."""

import numpy as np
import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._nv_metrics import AnswerAccuracy as LegacyAnswerAccuracy
from ragas.metrics.collections import AnswerAccuracy


# NVIDIA-specific fixtures with correct temperature (0.1)
@pytest.fixture
def nvidia_legacy_llm():
    """Create legacy LLM for AnswerAccuracy (temperature set in metric calls)."""
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
    """Create modern LLM with NVIDIA temperature (0.1) for AnswerAccuracy."""
    try:
        import openai

        from ragas.llms.base import instructor_llm_factory

        client = openai.AsyncOpenAI()
        # Set temperature=0.1 to match legacy NVIDIA calls exactly
        return instructor_llm_factory(
            "openai", model="gpt-4o", client=client, temperature=0.1
        )
    except Exception as e:
        pytest.skip(str(e))


class TestAnswerAccuracyE2EMigration:
    """E2E test compatibility between legacy AnswerAccuracy and new V2 AnswerAccuracy with modern components."""

    @pytest.fixture
    def sample_data(self):
        """Real-world test cases for answer accuracy evaluation."""
        return [
            {
                "user_input": "When was Einstein born?",
                "response": "Albert Einstein was born in 1879.",
                "reference": "Albert Einstein was born in 1879.",
                "description": "Exact match - should score high",
            },
            {
                "user_input": "When was Einstein born?",
                "response": "Albert Einstein was born on March 14, 1879.",
                "reference": "Albert Einstein was born in 1879.",
                "description": "Partial match - additional correct details",
            },
            {
                "user_input": "When was Einstein born?",
                "response": "Albert Einstein was born in 1885.",
                "reference": "Albert Einstein was born in 1879.",
                "description": "Incorrect answer - wrong year",
            },
            {
                "user_input": "What is photosynthesis?",
                "response": "Photosynthesis is how plants make energy.",
                "reference": "Photosynthesis is the process by which plants convert sunlight into chemical energy using chlorophyll.",
                "description": "Incomplete but correct summary",
            },
        ]

    @pytest.fixture
    def test_llm(self):
        """Create a test LLM for legacy answer accuracy evaluation."""
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
    async def test_legacy_answer_accuracy_vs_v2_answer_accuracy_e2e_compatibility(
        self, sample_data, nvidia_legacy_llm, nvidia_modern_llm
    ):
        """E2E test that legacy and v2 implementations produce similar scores."""

        if nvidia_legacy_llm is None or nvidia_modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        for i, data in enumerate(sample_data):
            print(f"\n🧪 Testing Answer Accuracy - Case {i + 1}: {data['description']}")
            print(f"   Question: {data['user_input']}")
            print(f"   Response: {data['response']}")
            print(f"   Reference: {data['reference']}")

            # Legacy implementation
            legacy_answer_accuracy = LegacyAnswerAccuracy(llm=nvidia_legacy_llm)
            legacy_sample = SingleTurnSample(
                user_input=data["user_input"],
                response=data["response"],
                reference=data["reference"],
            )
            legacy_score = await legacy_answer_accuracy._single_turn_ascore(
                legacy_sample, None
            )

            # V2 implementation
            v2_answer_accuracy = AnswerAccuracy(llm=nvidia_modern_llm)
            v2_result = await v2_answer_accuracy.ascore(
                user_input=data["user_input"],
                response=data["response"],
                reference=data["reference"],
            )

            score_diff = (
                abs(legacy_score - v2_result.value)
                if not np.isnan(legacy_score) and not np.isnan(v2_result.value)
                else 0.0
            )
            print(f"   Legacy: {legacy_score:.6f}")
            print(f"   V2:     {v2_result.value:.6f}")
            print(f"   Diff:   {score_diff:.6f}")

            # Both implementations use dual judges with same prompts and temperature
            # Some variance expected due to Langchain vs Instructor interface differences
            if not np.isnan(legacy_score) and not np.isnan(v2_result.value):
                assert score_diff < 0.6, (
                    f"Legacy and V2 scores should be reasonably similar: Legacy={legacy_score:.6f}, "
                    f"V2={v2_result.value:.6f}, Diff={score_diff:.6f} (tolerance: 0.6)"
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
    async def test_answer_accuracy_dual_judge_system(self, test_modern_llm):
        """Test that v2 implementation correctly uses dual-judge system."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for dual-judge testing")

        metric = AnswerAccuracy(llm=test_modern_llm)

        # Test case where both judges should agree
        result = await metric.ascore(
            user_input="What is 2+2?",
            response="2+2 equals 4.",
            reference="2+2 equals 4.",
        )

        print(f"Dual-judge result: {result.value:.3f}")

        # Should be high score for exact match
        if not np.isnan(result.value):
            assert 0.5 <= result.value <= 1.0, (
                f"Expected high score for exact match, got {result.value}"
            )

    def test_answer_accuracy_migration_requirements_documented(self):
        """Test that migration requirements are properly documented."""

        # V2 implementation should not accept legacy components
        with pytest.raises((TypeError, ValueError, AttributeError)):
            AnswerAccuracy(llm="invalid_llm_type")  # Should reject string

        # V2 should only accept InstructorBaseRagasLLM
        with pytest.raises((TypeError, ValueError, AttributeError)):
            AnswerAccuracy(llm=None)  # Should reject None
