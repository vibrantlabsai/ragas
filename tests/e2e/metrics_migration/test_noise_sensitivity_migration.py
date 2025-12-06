"""E2E tests for Noise Sensitivity metric migration from v1 to v2."""

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._noise_sensitivity import NoiseSensitivity as LegacyNoiseSensitivity
from ragas.metrics.collections import NoiseSensitivity


class TestNoiseSensitivityE2EMigration:
    """E2E test compatibility between legacy NoiseSensitivity and new V2 NoiseSensitivity with modern components."""

    @pytest.fixture
    def sample_data(self):
        """Real-world test cases for noise sensitivity evaluation."""
        return [
            {
                "user_input": "What is the Life Insurance Corporation of India (LIC) known for?",
                "response": "The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.",
                "reference": "The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.",
                "retrieved_contexts": [
                    "The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.",
                    "LIC is the largest insurance company in India, with a vast network of policyholders and huge investments.",
                    "As the largest institutional investor in India, LIC manages substantial funds, contributing to the financial stability of the country.",
                    "The Indian economy is one of the fastest-growing major economies in the world, thanks to sectors like finance, technology, manufacturing etc.",
                ],
                "description": "Complex case with relevant and irrelevant contexts",
            },
            {
                "user_input": "What is photosynthesis?",
                "response": "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll.",
                "reference": "Photosynthesis is the process by which plants use sunlight, carbon dioxide, and water to produce glucose and oxygen using chlorophyll.",
                "retrieved_contexts": [
                    "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
                    "Plants use chlorophyll to capture sunlight for photosynthesis.",
                    "Albert Einstein developed the theory of relativity.",
                ],
                "description": "Simple case with clear relevant/irrelevant split",
            },
        ]

    @pytest.fixture
    def test_llm(self):
        """Create a LangChain LLM for legacy noise sensitivity evaluation."""
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
    async def test_legacy_noise_sensitivity_vs_v2_noise_sensitivity_e2e_compatibility(
        self, sample_data, test_llm, test_modern_llm
    ):
        """E2E test that legacy and v2 implementations produce similar scores."""

        if test_llm is None or test_modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        # Test both relevant and irrelevant modes
        modes = ["relevant", "irrelevant"]

        for mode in modes:
            print(f"\n🧪 Testing Noise Sensitivity - Mode: {mode}")
            print("-" * 50)

            for i, data in enumerate(sample_data):
                print(f"\n📋 Case {i + 1}: {data['description']}")
                print(f"   Question: {data['user_input'][:60]}...")
                print(f"   Response: {data['response'][:60]}...")
                print(f"   Contexts: {len(data['retrieved_contexts'])} contexts")

                # Legacy implementation
                legacy_noise_sensitivity = LegacyNoiseSensitivity(
                    llm=test_llm, mode=mode
                )
                legacy_sample = SingleTurnSample(
                    user_input=data["user_input"],
                    response=data["response"],
                    reference=data["reference"],
                    retrieved_contexts=data["retrieved_contexts"],
                )
                legacy_score = await legacy_noise_sensitivity._single_turn_ascore(
                    legacy_sample, None
                )

                # V2 implementation
                v2_noise_sensitivity = NoiseSensitivity(llm=test_modern_llm, mode=mode)
                v2_result = await v2_noise_sensitivity.ascore(
                    user_input=data["user_input"],
                    response=data["response"],
                    reference=data["reference"],
                    retrieved_contexts=data["retrieved_contexts"],
                )

                score_diff = abs(legacy_score - v2_result.value)
                print(f"   Legacy: {legacy_score:.6f}")
                print(f"   V2:     {v2_result.value:.6f}")
                print(f"   Diff:   {score_diff:.6f}")

                # Ensure implementations give reasonably similar scores
                # Complex multi-step metric may have some variance
                assert score_diff < 0.3, (
                    f"Legacy and V2 scores should be reasonably similar: Legacy={legacy_score:.6f}, "
                    f"V2={v2_result.value:.6f}, Diff={score_diff:.6f} (tolerance: 0.3)"
                )
                print("   ✅ Both implementations give consistent scores")

                # Validate score ranges
                assert 0.0 <= legacy_score <= 1.0
                assert 0.0 <= v2_result.value <= 1.0

    @pytest.mark.asyncio
    async def test_noise_sensitivity_mode_configuration(self, test_modern_llm):
        """Test that v2 implementation respects mode configuration."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for mode testing")

        # Test data with clear relevant/irrelevant split
        test_case = {
            "user_input": "What is photosynthesis?",
            "response": "Photosynthesis converts sunlight to energy.",
            "reference": "Photosynthesis is the process by which plants convert sunlight into energy.",
            "retrieved_contexts": [
                "Plants use photosynthesis to convert light into energy.",  # Relevant
                "Albert Einstein developed relativity theory.",  # Irrelevant
            ],
        }

        # Test relevant mode
        relevant_metric = NoiseSensitivity(llm=test_modern_llm, mode="relevant")
        relevant_result = await relevant_metric.ascore(**test_case)

        # Test irrelevant mode
        irrelevant_metric = NoiseSensitivity(llm=test_modern_llm, mode="irrelevant")
        irrelevant_result = await irrelevant_metric.ascore(**test_case)

        print(f"Relevant mode score: {relevant_result.value:.3f}")
        print(f"Irrelevant mode score: {irrelevant_result.value:.3f}")

        # Validate score ranges
        assert 0.0 <= relevant_result.value <= 1.0
        assert 0.0 <= irrelevant_result.value <= 1.0

        # Different modes should potentially produce different scores
        # (though they might be the same for some data)

    @pytest.mark.asyncio
    async def test_noise_sensitivity_parameter_validation(self, test_modern_llm):
        """Test that v2 implementation validates parameters correctly."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for parameter testing")

        # Test invalid mode
        with pytest.raises(ValueError, match="Invalid argument passed for 'mode'"):
            NoiseSensitivity(llm=test_modern_llm, mode="invalid_mode")

        # Test valid modes
        relevant_metric = NoiseSensitivity(llm=test_modern_llm, mode="relevant")
        irrelevant_metric = NoiseSensitivity(llm=test_modern_llm, mode="irrelevant")

        assert relevant_metric.mode == "relevant"
        assert irrelevant_metric.mode == "irrelevant"

    def test_noise_sensitivity_migration_requirements_documented(self):
        """Test that migration requirements are properly documented."""

        # V2 implementation should not accept legacy components
        with pytest.raises((TypeError, ValueError, AttributeError)):
            NoiseSensitivity(llm="invalid_llm_type")  # Should reject string

        # V2 should only accept InstructorBaseRagasLLM
        with pytest.raises((TypeError, ValueError, AttributeError)):
            NoiseSensitivity(llm=None)  # Should reject None
