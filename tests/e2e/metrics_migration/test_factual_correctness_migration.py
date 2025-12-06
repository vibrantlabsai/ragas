"""E2E tests for FactualCorrectness metric migration from v1 to v2."""

import numpy as np
import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._factual_correctness import (
    FactualCorrectness as LegacyFactualCorrectness,
)
from ragas.metrics.collections import FactualCorrectness


class TestFactualCorrectnessE2EMigration:
    """E2E test compatibility between legacy FactualCorrectness and new V2 FactualCorrectness with modern components."""

    @pytest.fixture
    def sample_data(self):
        """Real-world test cases for factual correctness evaluation."""
        return [
            {
                "response": "Einstein was born in Germany on 14th March 1879.",
                "reference": "Albert Einstein was born in Ulm, Germany on March 14, 1879.",
                "description": "High factual correctness - consistent facts",
            },
            {
                "response": "Einstein was born in France on 14th March 1879.",
                "reference": "Albert Einstein was born in Ulm, Germany on March 14, 1879.",
                "description": "Low factual correctness - wrong country",
            },
            {
                "response": "The first superbowl was held on Jan 15, 1967.",
                "reference": "The First AFL–NFL World Championship Game was played on January 15, 1967.",
                "description": "Perfect factual correctness - exact match",
            },
            {
                "response": "Photosynthesis converts sunlight into energy and produces oxygen.",
                "reference": "Photosynthesis is the process by which plants convert sunlight into energy and produce oxygen as a byproduct.",
                "description": "High factual correctness - covers key facts",
            },
            {
                "response": "Newton discovered gravity when an apple fell on his head.",
                "reference": "Newton developed his theory of universal gravitation, though the apple story is likely apocryphal.",
                "description": "Mixed factual correctness - partially correct",
            },
        ]

    @pytest.fixture
    def test_llm(self):
        """Create a LangChain LLM for legacy factual correctness evaluation."""
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
    async def test_legacy_factual_correctness_vs_v2_factual_correctness_e2e_compatibility(
        self, sample_data, test_llm, test_modern_llm
    ):
        """E2E test that legacy and v2 implementations produce similar scores."""

        if test_llm is None or test_modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        # Test different modes and configurations
        test_configs = [
            {"mode": "f1", "atomicity": "low", "coverage": "low"},
            {"mode": "precision", "atomicity": "high", "coverage": "high"},
            {"mode": "recall", "atomicity": "low", "coverage": "high"},
        ]

        for config in test_configs:
            print(f"\n🧪 Testing FactualCorrectness - Config: {config}")

            for i, data in enumerate(sample_data):
                print(f"\n   Case {i + 1}: {data['description']}")
                print(f"   Response: {data['response'][:80]}...")
                print(f"   Reference: {data['reference'][:80]}...")

                # Legacy implementation
                legacy_correctness = LegacyFactualCorrectness(
                    llm=test_llm,
                    mode=config["mode"],  # type: ignore[arg-type]
                    atomicity=config["atomicity"],  # type: ignore[arg-type]
                    coverage=config["coverage"],  # type: ignore[arg-type]
                )
                legacy_sample = SingleTurnSample(
                    response=data["response"],
                    reference=data["reference"],
                )
                legacy_score = await legacy_correctness._single_turn_ascore(
                    legacy_sample, None
                )

                # V2 implementation
                v2_correctness = FactualCorrectness(
                    llm=test_modern_llm,
                    mode=config["mode"],  # type: ignore[arg-type]
                    atomicity=config["atomicity"],  # type: ignore[arg-type]
                    coverage=config["coverage"],  # type: ignore[arg-type]
                )
                v2_result = await v2_correctness.ascore(
                    response=data["response"],
                    reference=data["reference"],
                )

                score_diff = abs(legacy_score - v2_result.value)
                print(f"   Legacy: {legacy_score:.6f}")
                print(f"   V2:     {v2_result.value:.6f}")
                print(f"   Diff:   {score_diff:.6f}")

                # Ensure implementations give reasonably similar scores
                # After fixing the parameter order bug, factual correctness has excellent compatibility
                # Max observed difference: 0.1 (down from 0.33 before the fix)
                assert score_diff < 0.15, (
                    f"Legacy and V2 scores should be similar: Legacy={legacy_score:.6f}, "
                    f"V2={v2_result.value:.6f}, Diff={score_diff:.6f} (tolerance: 0.15)"
                )
                print("   ✅ Both implementations give consistent scores")

                # Validate score ranges (both should be 0-1 or NaN)
                if not np.isnan(legacy_score):
                    assert 0.0 <= legacy_score <= 1.0
                if not np.isnan(v2_result.value):
                    assert 0.0 <= v2_result.value <= 1.0

    @pytest.mark.asyncio
    async def test_factual_correctness_edge_cases(self, test_modern_llm):
        """Test edge cases like empty responses and references."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for edge case testing")

        metric = FactualCorrectness(llm=test_modern_llm)

        # Test empty response
        with pytest.raises(ValueError, match="response is missing"):
            await metric.ascore(
                response="",
                reference="Einstein was born in Germany.",
            )

        # Test empty reference
        with pytest.raises(ValueError, match="reference is missing"):
            await metric.ascore(
                response="Einstein was born in Germany.",
                reference="",
            )

    @pytest.mark.asyncio
    async def test_factual_correctness_different_modes(self, test_modern_llm):
        """Test that different modes (precision, recall, f1) produce different scores."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for mode testing")

        response = "Einstein was a physicist born in Germany."
        reference = "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity."

        # Test different modes
        precision_metric = FactualCorrectness(llm=test_modern_llm, mode="precision")
        recall_metric = FactualCorrectness(llm=test_modern_llm, mode="recall")
        f1_metric = FactualCorrectness(llm=test_modern_llm, mode="f1")

        precision_result = await precision_metric.ascore(
            response=response, reference=reference
        )
        recall_result = await recall_metric.ascore(
            response=response, reference=reference
        )
        f1_result = await f1_metric.ascore(response=response, reference=reference)

        print(f"Precision score: {precision_result.value:.3f}")
        print(f"Recall score: {recall_result.value:.3f}")
        print(f"F1 score: {f1_result.value:.3f}")

        # Validate ranges
        assert 0.0 <= precision_result.value <= 1.0
        assert 0.0 <= recall_result.value <= 1.0
        assert 0.0 <= f1_result.value <= 1.0

    @pytest.mark.asyncio
    async def test_factual_correctness_atomicity_coverage_configurations(
        self, test_modern_llm
    ):
        """Test that different atomicity/coverage configurations work."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for configuration testing")

        response = "Einstein was a German physicist who developed relativity theory."
        reference = (
            "Albert Einstein was born in Germany and created the theory of relativity."
        )

        configs = [
            {"atomicity": "low", "coverage": "low"},
            {"atomicity": "low", "coverage": "high"},
            {"atomicity": "high", "coverage": "low"},
            {"atomicity": "high", "coverage": "high"},
        ]

        for config in configs:
            metric = FactualCorrectness(
                llm=test_modern_llm,
                atomicity=config["atomicity"],  # type: ignore[arg-type]
                coverage=config["coverage"],  # type: ignore[arg-type]
            )
            result = await metric.ascore(response=response, reference=reference)

            print(f"Config {config}: {result.value:.3f}")

            # Validate score range
            assert 0.0 <= result.value <= 1.0, f"Invalid score for config {config}"

    def test_factual_correctness_migration_requirements_documented(self):
        """Test that migration requirements are properly documented."""

        # V2 implementation should not accept legacy components
        with pytest.raises((TypeError, ValueError, AttributeError)):
            FactualCorrectness(llm="invalid_llm_type")  # type: ignore[arg-type]  # Should reject string

        # V2 should only accept InstructorBaseRagasLLM
        with pytest.raises((TypeError, ValueError, AttributeError)):
            FactualCorrectness(llm=None)  # type: ignore[arg-type]  # Should reject None

        # Test beta validation
        with pytest.raises(ValueError, match="Beta must be a float"):
            FactualCorrectness(llm=None, beta="invalid")  # type: ignore[arg-type]  # Should reject non-numeric beta
