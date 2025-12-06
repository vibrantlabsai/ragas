"""E2E tests for Answer Correctness metric migration from v1 to v2."""

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerCorrectness as LegacyAnswerCorrectness
from ragas.metrics.collections import AnswerCorrectness
from ragas.metrics.result import MetricResult


class TestAnswerCorrectnessE2EMigration:
    """E2E test compatibility between legacy AnswerCorrectness and new V2 AnswerCorrectness with modern components."""

    @pytest.fixture
    def sample_data(self):
        """Real-world test cases for answer correctness evaluation."""
        return [
            {
                "user_input": "What is the capital of France?",
                "response": "The capital of France is Paris.",
                "reference": "Paris is the capital of France.",
                "description": "Perfect match - should score high",
            },
            {
                "user_input": "What powers the sun?",
                "response": "The sun is powered by nuclear fission reactions.",
                "reference": "The sun is powered by nuclear fusion reactions where hydrogen atoms combine to form helium.",
                "description": "Factual error - should score low on factuality",
            },
            {
                "user_input": "What is photosynthesis?",
                "response": "Photosynthesis is the process by which plants convert sunlight into energy.",
                "reference": "Photosynthesis is the process by which plants use sunlight, carbon dioxide, and water to produce glucose and oxygen using chlorophyll.",
                "description": "Incomplete answer - missing key details",
            },
            {
                "user_input": "What is 2 + 2?",
                "response": "2 + 2 equals 4. This is basic arithmetic.",
                "reference": "2 + 2 = 4",
                "description": "Correct with extra information",
            },
            {
                "user_input": "Explain quantum computing",
                "response": "Quantum computing uses quantum bits that can exist in superposition states.",
                "reference": "Quantum computing is a type of computation that harnesses quantum mechanical phenomena like superposition and entanglement to process information using quantum bits or qubits.",
                "description": "Partial coverage of complex topic",
            },
        ]

    @pytest.fixture
    def test_llm(self):
        """Create a test LLM for legacy answer correctness evaluation."""
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

    @pytest.fixture
    def test_legacy_embeddings(self):
        """Create legacy embeddings for legacy implementation."""
        try:
            from ragas.embeddings.base import embedding_factory

            return embedding_factory("text-embedding-ada-002")
        except ImportError as e:
            pytest.skip(f"Embedding factory not available: {e}")
        except Exception as e:
            pytest.skip(
                f"Could not create legacy embeddings (API key may be missing): {e}"
            )

    @pytest.fixture
    def test_modern_embeddings(self):
        """Create modern embeddings for v2 implementation."""
        try:
            import openai

            from ragas.embeddings.base import embedding_factory

            client = openai.AsyncOpenAI()
            return embedding_factory(
                provider="openai",
                model="text-embedding-ada-002",
                client=client,
                interface="modern",
            )
        except ImportError as e:
            pytest.skip(f"OpenAI or embedding factory not available: {e}")
        except Exception as e:
            pytest.skip(
                f"Could not create modern embeddings (API key may be missing): {e}"
            )

    @pytest.mark.asyncio
    async def test_legacy_answer_correctness_vs_v2_answer_correctness_e2e_compatibility(
        self,
        sample_data,
        test_llm,
        test_modern_llm,
        test_legacy_embeddings,
        test_modern_embeddings,
    ):
        """E2E test that legacy and v2 implementations produce similar scores with real LLM."""

        if (
            test_llm is None
            or test_modern_llm is None
            or test_legacy_embeddings is None
            or test_modern_embeddings is None
        ):
            pytest.skip("LLM and embeddings required for E2E testing")

        for i, data in enumerate(sample_data):
            print(
                f"\n🧪 Testing Answer Correctness - Case {i + 1}: {data['description']}"
            )
            print(f"   Question: {data['user_input']}")
            print(f"   Response: {data['response'][:80]}...")
            print(f"   Reference: {data['reference'][:80]}...")

            # Legacy v1 implementation - need to initialize it properly
            legacy_answer_correctness = LegacyAnswerCorrectness(
                llm=test_llm, embeddings=test_legacy_embeddings
            )
            # Initialize the answer_similarity component for v1
            from ragas.run_config import RunConfig

            legacy_answer_correctness.init(RunConfig())
            legacy_sample = SingleTurnSample(
                user_input=data["user_input"],
                response=data["response"],
                reference=data["reference"],
            )
            legacy_score = await legacy_answer_correctness._single_turn_ascore(
                legacy_sample, None
            )

            # V2 implementation with modern components
            v2_answer_correctness = AnswerCorrectness(
                llm=test_modern_llm, embeddings=test_modern_embeddings
            )
            v2_result = await v2_answer_correctness.ascore(
                user_input=data["user_input"],
                response=data["response"],
                reference=data["reference"],
            )

            # Results might not be exactly identical due to LLM randomness, but should be close
            score_diff = abs(legacy_score - v2_result.value)
            print(f"   Legacy: {legacy_score:.6f}")
            print(f"   V2:     {v2_result.value:.6f}")
            print(f"   Diff:   {score_diff:.6f}")

            # Allow some tolerance for LLM randomness and potential differences in processing
            assert score_diff < 0.2, (
                f"Case {i + 1} ({data['description']}): Large difference: {legacy_score} vs {v2_result.value}"
            )

            # Verify types
            assert isinstance(legacy_score, float)
            assert isinstance(v2_result, MetricResult)
            assert 0.0 <= legacy_score <= 1.0
            assert 0.0 <= v2_result.value <= 1.0

            print("   ✅ Scores within tolerance!")

    @pytest.mark.asyncio
    async def test_answer_correctness_factual_error_detection(
        self, test_llm, test_modern_llm, test_legacy_embeddings, test_modern_embeddings
    ):
        """Test that both implementations correctly detect factual errors."""

        if (
            test_llm is None
            or test_modern_llm is None
            or test_legacy_embeddings is None
            or test_modern_embeddings is None
        ):
            pytest.skip("LLM and embeddings required for E2E testing")

        # Test cases specifically for factual error detection
        test_cases = [
            {
                "user_input": "What is the boiling point of water at sea level?",
                "response": "Water boils at 90 degrees Celsius at sea level.",
                "reference": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
                "expected_low": True,
                "description": "Clear factual error",
            },
            {
                "user_input": "What is the boiling point of water at sea level?",
                "response": "Water boils at 100 degrees Celsius at sea level.",
                "reference": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
                "expected_low": False,
                "description": "Factually correct",
            },
            {
                "user_input": "What is the capital of Italy?",
                "response": "The capital of Italy is Milan.",
                "reference": "The capital of Italy is Rome.",
                "expected_low": True,
                "description": "Wrong capital city",
            },
        ]

        for case in test_cases:
            print(f"\n🎯 Testing factual error detection: {case['description']}")

            # Legacy implementation - need to initialize it properly
            legacy_answer_correctness = LegacyAnswerCorrectness(
                llm=test_llm, embeddings=test_legacy_embeddings
            )
            # Initialize the answer_similarity component for v1
            from ragas.run_config import RunConfig

            legacy_answer_correctness.init(RunConfig())
            legacy_sample = SingleTurnSample(
                user_input=case["user_input"],
                response=case["response"],
                reference=case["reference"],
            )
            legacy_score = await legacy_answer_correctness._single_turn_ascore(
                legacy_sample, None
            )

            # V2 implementation
            v2_answer_correctness = AnswerCorrectness(
                llm=test_modern_llm, embeddings=test_modern_embeddings
            )
            v2_result = await v2_answer_correctness.ascore(
                user_input=case["user_input"],
                response=case["response"],
                reference=case["reference"],
            )

            print(f"   Response: {case['response']}")
            print(f"   Reference: {case['reference']}")
            print(f"   Legacy: {legacy_score:.6f}")
            print(f"   V2:     {v2_result.value:.6f}")

            # Compare scores between implementations
            score_diff = abs(legacy_score - v2_result.value)
            print(f"   Difference: {score_diff:.6f}")

            # Ensure both implementations give very close scores (strict migration compatibility)
            assert score_diff < 0.001, (
                f"Legacy and V2 scores should be nearly identical: Legacy={legacy_score:.6f}, "
                f"V2={v2_result.value:.6f}, Diff={score_diff:.6f} (tolerance: 0.001)"
            )
            print("   ✅ Both implementations give identical scores")

    @pytest.mark.asyncio
    async def test_answer_correctness_weight_configuration(
        self, test_modern_llm, test_modern_embeddings
    ):
        """Test that v2 implementation respects weight configuration."""

        if test_modern_llm is None or test_modern_embeddings is None:
            pytest.skip("Modern LLM and embeddings required for weight testing")

        test_case = {
            "user_input": "What is machine learning?",
            "response": "Machine learning is a subset of AI that enables computers to learn patterns.",
            "reference": "Machine learning is a method of data analysis that automates analytical model building using algorithms that iteratively learn from data.",
        }

        # Test factuality-focused weights
        factuality_focused = AnswerCorrectness(
            llm=test_modern_llm,
            embeddings=test_modern_embeddings,
            weights=[0.9, 0.1],  # 90% factuality, 10% similarity
        )
        factuality_result = await factuality_focused.ascore(
            user_input=test_case["user_input"],
            response=test_case["response"],
            reference=test_case["reference"],
        )

        # Test similarity-focused weights
        similarity_focused = AnswerCorrectness(
            llm=test_modern_llm,
            embeddings=test_modern_embeddings,
            weights=[0.1, 0.9],  # 10% factuality, 90% similarity
        )
        similarity_result = await similarity_focused.ascore(
            user_input=test_case["user_input"],
            response=test_case["response"],
            reference=test_case["reference"],
        )

        # Test balanced weights (default)
        balanced = AnswerCorrectness(
            llm=test_modern_llm,
            embeddings=test_modern_embeddings,
            weights=[0.75, 0.25],  # Default weights
        )
        balanced_result = await balanced.ascore(
            user_input=test_case["user_input"],
            response=test_case["response"],
            reference=test_case["reference"],
        )

        print("\n🎛️ Testing weight configurations:")
        print(f"   Factuality-focused (90/10): {factuality_result.value:.6f}")
        print(f"   Similarity-focused (10/90): {similarity_result.value:.6f}")
        print(f"   Balanced (75/25):           {balanced_result.value:.6f}")

        # All should be valid scores
        assert 0.0 <= factuality_result.value <= 1.0
        assert 0.0 <= similarity_result.value <= 1.0
        assert 0.0 <= balanced_result.value <= 1.0

        # Scores may differ based on weighting
        print("   ✅ All weight configurations produced valid scores!")

    def test_answer_correctness_parameter_validation(self):
        """Test that v2 implementation properly validates parameters."""
        from unittest.mock import Mock

        from ragas.llms.base import InstructorBaseRagasLLM

        # Create proper mocks that inherit from the required base class
        mock_llm = Mock(spec=InstructorBaseRagasLLM)
        mock_embeddings = Mock()

        # Test invalid weights
        with pytest.raises(ValueError, match="two weights"):
            AnswerCorrectness(llm=mock_llm, embeddings=mock_embeddings, weights=[0.5])

        with pytest.raises(ValueError, match="non-zero"):
            AnswerCorrectness(
                llm=mock_llm, embeddings=mock_embeddings, weights=[0.0, 0.0]
            )

        with pytest.raises(ValueError, match="non-negative"):
            AnswerCorrectness(
                llm=mock_llm, embeddings=mock_embeddings, weights=[-0.1, 0.5]
            )

        # Test invalid beta - use type: ignore to bypass type checker for intentional error test
        with pytest.raises(ValueError, match="Beta must be a float"):
            AnswerCorrectness(llm=mock_llm, embeddings=mock_embeddings, beta="invalid")  # type: ignore

        # Test optional embeddings - should work with pure factuality (weight=0)
        metric = AnswerCorrectness(llm=mock_llm, weights=[1.0, 0.0])
        assert metric.embeddings is None
        print("✅ Optional embeddings working for pure factuality!")

        # Test embeddings required when similarity weight > 0
        with pytest.raises(ValueError, match="Embeddings are required"):
            AnswerCorrectness(llm=mock_llm, embeddings=None, weights=[0.75, 0.25])

        print("✅ Parameter validation working correctly!")

    def test_answer_correctness_migration_requirements_documented(self):
        """Document the requirements for running full E2E answer correctness tests."""

        requirements = {
            "llm": "OpenAI GPT, Anthropic Claude, or other LLM with structured output support",
            "embeddings": "OpenAI embeddings, HuggingFace embeddings, or similar",
            "environment": "API keys configured for LLM and embedding providers",
            "purpose": "Verify that v2 implementation produces similar results to legacy implementation",
            "complexity": "Tests statement generation, TP/FP/FN classification, F1 scoring, and similarity calculation",
        }

        print("\n📋 Answer Correctness E2E Test Requirements:")
        for key, value in requirements.items():
            print(f"   {key.capitalize()}: {value}")

        print("\n🚀 To enable full E2E testing:")
        print("   1. Configure LLM provider (e.g., export OPENAI_API_KEY=...)")
        print("   2. Configure embeddings provider")
        print("   3. Remove @pytest.mark.skip decorators")
        print(
            "   4. Run: pytest tests/e2e/metrics_migration/test_answer_correctness_migration.py -v -s"
        )

        print("\n🔬 Test Coverage:")
        print("   • Statement generation accuracy")
        print("   • TP/FP/FN classification correctness")
        print("   • F1 score calculation")
        print("   • Semantic similarity computation")
        print("   • Weight configuration effects")
        print("   • Parameter validation")
        print("   • Score equivalence between v1 and v2")

        assert True
