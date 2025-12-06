"""E2E tests for Summary Score metric migration from v1 to v2."""

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._summarization import SummarizationScore as LegacySummaryScore
from ragas.metrics.collections import SummaryScore


class TestSummaryScoreE2EMigration:
    """E2E test compatibility between legacy SummaryScore and new V2 SummaryScore with modern components."""

    @pytest.fixture
    def sample_data(self):
        """Real-world test cases for summary score evaluation."""
        return [
            {
                "reference_contexts": [
                    "Apple Inc. is a technology company based in Cupertino, California. Founded by Steve Jobs in 1976, it reached a market capitalization of $3 trillion in 2023. The company is known for innovative products like iPhone, iPad, and Mac computers. Apple has retail stores worldwide and employs over 150,000 people."
                ],
                "response": "Apple Inc. is a technology company founded by Steve Jobs in 1976, based in Cupertino, California. The company reached a $3 trillion market cap in 2023.",
                "description": "Good summary with key facts",
            },
            {
                "reference_contexts": [
                    "Climate change refers to long-term shifts in global temperatures and weather patterns. Since the 1800s, human activities have been the main driver of climate change, primarily due to fossil fuel burning which releases greenhouse gases. The effects include rising sea levels, extreme weather events, and ecosystem disruption."
                ],
                "response": "Weather changes happen sometimes.",
                "description": "Very brief summary missing key details",
            },
            {
                "reference_contexts": [
                    "The Great Wall of China is an ancient series of walls and fortifications built across the northern borders of China. Construction began in the 7th century BC and continued for centuries. The wall stretches over 13,000 miles and was built to protect against invasions."
                ],
                "response": "The Great Wall of China is an ancient series of walls and fortifications built across northern China starting in the 7th century BC. It stretches over 13,000 miles and was built for protection against invasions.",
                "description": "Comprehensive summary with most details",
            },
        ]

    @pytest.fixture
    def test_llm(self):
        """Create a LangChain LLM for legacy summary score evaluation."""
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
    async def test_legacy_summary_score_vs_v2_summary_score_e2e_compatibility(
        self, sample_data, test_llm, test_modern_llm
    ):
        """E2E test that legacy and v2 implementations produce similar scores."""

        if test_llm is None or test_modern_llm is None:
            pytest.skip("LLM required for E2E testing")

        for i, data in enumerate(sample_data):
            print(f"\n🧪 Testing Summary Score - Case {i + 1}: {data['description']}")
            print(f"   Contexts: {data['reference_contexts'][0][:80]}...")
            print(f"   Response: {data['response'][:80]}...")

            # Legacy implementation
            legacy_summary_score = LegacySummaryScore(llm=test_llm)
            legacy_sample = SingleTurnSample(
                reference_contexts=data["reference_contexts"],
                response=data["response"],
            )
            legacy_score = await legacy_summary_score._single_turn_ascore(
                legacy_sample, None
            )

            # V2 implementation
            v2_summary_score = SummaryScore(llm=test_modern_llm)
            v2_result = await v2_summary_score.ascore(
                reference_contexts=data["reference_contexts"],
                response=data["response"],
            )

            score_diff = abs(legacy_score - v2_result.value)
            print(f"   Legacy: {legacy_score:.6f}")
            print(f"   V2:     {v2_result.value:.6f}")
            print(f"   Diff:   {score_diff:.6f}")

            # Ensure implementations give reasonably similar scores for complex multi-step metric
            assert score_diff < 0.2, (
                f"Legacy and V2 scores should be reasonably similar: Legacy={legacy_score:.6f}, "
                f"V2={v2_result.value:.6f}, Diff={score_diff:.6f} (tolerance: 0.2)"
            )
            print("   ✅ Both implementations give consistent scores")

            # Validate score ranges
            assert 0.0 <= legacy_score <= 1.0
            assert 0.0 <= v2_result.value <= 1.0

    @pytest.mark.asyncio
    async def test_summary_score_weight_configuration(self, test_modern_llm):
        """Test that v2 implementation respects weight configuration."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for weight testing")

        # Test data
        contexts = [
            "Apple Inc. is a technology company founded by Steve Jobs in 1976. The company is based in Cupertino, California."
        ]
        summary = "Apple is a tech company."

        # Test different coefficient values
        coefficients = [0.0, 0.5, 1.0]  # 0=only QA, 0.5=balanced, 1.0=only conciseness

        results = []
        for coeff in coefficients:
            metric = SummaryScore(llm=test_modern_llm, coeff=coeff, length_penalty=True)
            result = await metric.ascore(reference_contexts=contexts, response=summary)
            results.append(result.value)

            # Validate score range
            assert 0.0 <= result.value <= 1.0

        print(
            f"Coefficient results: coeff=0.0: {results[0]:.3f}, coeff=0.5: {results[1]:.3f}, coeff=1.0: {results[2]:.3f}"
        )

        # Different coefficients should produce different scores
        assert results[0] != results[2], (
            "Different coefficients should produce different scores"
        )

    @pytest.mark.asyncio
    async def test_summary_score_parameter_validation(self, test_modern_llm):
        """Test that v2 implementation validates parameters correctly."""

        if test_modern_llm is None:
            pytest.skip("Modern LLM required for parameter testing")

        # Test invalid coefficient (too high)
        with pytest.raises(ValueError, match="Coefficient must be between 0.0 and 1.0"):
            SummaryScore(llm=test_modern_llm, coeff=1.5)

        # Test invalid coefficient (negative)
        with pytest.raises(ValueError, match="Coefficient must be between 0.0 and 1.0"):
            SummaryScore(llm=test_modern_llm, coeff=-0.1)

        # Test valid configurations
        metric1 = SummaryScore(llm=test_modern_llm, length_penalty=True, coeff=0.0)
        metric2 = SummaryScore(llm=test_modern_llm, length_penalty=False, coeff=1.0)

        assert metric1.length_penalty is True
        assert metric1.coeff == 0.0
        assert metric2.length_penalty is False
        assert metric2.coeff == 1.0

    def test_summary_score_migration_requirements_documented(self):
        """Test that migration requirements are properly documented."""

        # V2 implementation should not accept legacy components
        with pytest.raises((TypeError, ValueError, AttributeError)):
            SummaryScore(llm="invalid_llm_type")  # type: ignore[arg-type]  # Should reject string

        # V2 should only accept InstructorBaseRagasLLM
        with pytest.raises((TypeError, ValueError, AttributeError)):
            SummaryScore(llm=None)  # type: ignore[arg-type]  # Should reject None
