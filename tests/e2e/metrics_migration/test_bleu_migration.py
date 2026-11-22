"""E2E tests for BLEU score metric migration from v1 to v2."""

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore as LegacyBleuScore, MetricResult
from ragas.metrics.collections import BleuScore


class TestBleuE2EMigration:
    """E2E test compatibility between legacy BleuScore and new V2 implementations."""

    @pytest.fixture
    def sample_data(self):
        """Real-world sample reference and response texts for testing."""
        return [
            {
                "reference": "The cat sat on the mat. The dog ran in the park.",
                "response": "The cat sat on the mat. The dog ran in the park.",
                "description": "Exact match",
            },
            {
                "reference": "Python is a high-level programming language. It was created by Guido van Rossum.",
                "response": "Python is a programming language. It was developed by Guido van Rossum.",
                "description": "Similar content with paraphrasing",
            },
            {
                "reference": "Machine learning is a subset of artificial intelligence. It enables computers to learn from data.",
                "response": "Deep learning uses neural networks. It processes complex patterns in data.",
                "description": "Related but different content",
            },
            {
                "reference": "The capital of France is Paris.",
                "response": "Paris is the capital and largest city of France.",
                "description": "Reordered content",
            },
            {
                "reference": "",
                "response": "Some response text",
                "description": "Empty reference",
            },
            {
                "reference": "Some reference text",
                "response": "",
                "description": "Empty response",
            },
        ]

    @pytest.mark.asyncio
    async def test_legacy_vs_v2_class_e2e_compatibility(self, sample_data):
        """E2E test that legacy and v2 class implementations produce identical scores."""

        for i, data in enumerate(sample_data):
            print(f"\n🧪 Testing BLEU - Case {i + 1}: {data['description']}")
            print(f"   Reference: {data['reference'][:50]}...")
            print(f"   Response:  {data['response'][:50]}...")

            legacy_bleu = LegacyBleuScore()
            legacy_sample = SingleTurnSample(
                user_input="dummy",
                response=data["response"],
                reference=data["reference"],
            )
            legacy_score = await legacy_bleu._single_turn_ascore(legacy_sample, None)

            v2_class_metric = BleuScore()
            v2_class_result = await v2_class_metric.ascore(
                reference=data["reference"],
                response=data["response"],
            )

            class_diff = abs(legacy_score - v2_class_result.value)

            print(f"   Legacy:      {legacy_score:.6f}")
            print(f"   V2 Class:    {v2_class_result.value:.6f}")
            print(f"   Diff:        {class_diff:.10f}")

            assert class_diff < 1e-10, (
                f"Case {i + 1} ({data['description']}): BLEU mismatch: "
                f"{legacy_score} != {v2_class_result.value}"
            )

            assert isinstance(legacy_score, float)
            assert isinstance(v2_class_result, MetricResult)

            print("   ✅ Legacy and V2 class produce identical scores!")

    @pytest.mark.asyncio
    async def test_bleu_score_performance_comparison(self, sample_data):
        """Compare performance characteristics between legacy and v2 class."""
        import time

        test_case = sample_data[0]

        print("\n⚡ Performance test: BLEU score")

        legacy_bleu = LegacyBleuScore()
        legacy_sample = SingleTurnSample(
            user_input="dummy",
            response=test_case["response"],
            reference=test_case["reference"],
        )

        start_time = time.time()
        legacy_score = await legacy_bleu._single_turn_ascore(legacy_sample, None)
        legacy_time = time.time() - start_time

        v2_class_metric = BleuScore()
        start_time = time.time()
        v2_class_result = await v2_class_metric.ascore(
            reference=test_case["reference"],
            response=test_case["response"],
        )
        v2_class_time = time.time() - start_time

        print(f"   Legacy:      {legacy_time:.4f}s → {legacy_score:.6f}")
        print(f"   V2 Class:    {v2_class_time:.4f}s → {v2_class_result.value:.6f}")

        assert abs(legacy_score - v2_class_result.value) < 1e-10
        assert isinstance(legacy_score, float)
        assert isinstance(v2_class_result, MetricResult)

    @pytest.mark.asyncio
    async def test_v2_class_no_components_needed(self):
        """Test that V2 class-based BleuScore doesn't require LLM or embeddings."""

        print("\n🔧 Testing V2 BleuScore component requirements:")

        metric = BleuScore()

        print(f"   has llm attr: {hasattr(metric, 'llm')}")
        print(f"   has embeddings attr: {hasattr(metric, 'embeddings')}")

        result = await metric.ascore(
            reference="The capital of France is Paris.",
            response="Paris is the capital of France.",
        )

        print(f"   Score: {result.value:.6f}")

        assert not hasattr(metric, "llm") or metric.__dict__.get("llm") is None
        assert (
            not hasattr(metric, "embeddings")
            or metric.__dict__.get("embeddings") is None
        )
        assert isinstance(result.value, float)
        assert 0.0 <= result.value <= 1.0

        print("   ✅ V2 BleuScore works without LLM/embeddings!")

    @pytest.mark.asyncio
    async def test_v2_class_batch_processing(self, sample_data):
        """Test V2 class-based BleuScore batch processing."""

        metric = BleuScore()

        batch_inputs = [
            {"reference": case["reference"], "response": case["response"]}
            for case in sample_data[:3]
        ]

        print(f"\n📦 Testing V2 class batch processing with {len(batch_inputs)} items:")

        results = await metric.abatch_score(batch_inputs)

        assert len(results) == len(batch_inputs)

        for i, (case, result) in enumerate(zip(sample_data[:3], results)):
            print(f"   Case {i + 1}: {result.value:.6f} - {case['description']}")
            assert isinstance(result.value, float)
            assert -1e-10 <= result.value <= 1.0 + 1e-10
            assert result.reason is None

        print("   ✅ V2 class batch processing works correctly!")

    @pytest.mark.asyncio
    async def test_bleu_with_custom_kwargs(self):
        """Test that custom kwargs are passed correctly to sacrebleu."""

        print("\n🔧 Testing BleuScore with custom kwargs:")

        metric_default = BleuScore()
        metric_custom = BleuScore(kwargs={"smooth_method": "exp"})

        reference = "The quick brown fox jumps over the lazy dog."
        response = "The quick brown fox jumps."

        result_default = await metric_default.ascore(
            reference=reference, response=response
        )
        result_custom = await metric_custom.ascore(
            reference=reference, response=response
        )

        print(f"   Default kwargs: {result_default.value:.6f}")
        print(f"   Custom kwargs:  {result_custom.value:.6f}")

        assert isinstance(result_default.value, float)
        assert isinstance(result_custom.value, float)
        assert 0.0 <= result_default.value <= 1.0
        assert 0.0 <= result_custom.value <= 1.0

        print("   ✅ Custom kwargs work correctly!")


    @pytest.mark.asyncio
    async def test_bleu_score_multi_sentence_reference(self):
        reference = "The cat sat. The dog ran. The bird exploded."
        response = "The cat sat. The dog ran. The bird flew away."

        legacy_score = await LegacyBleuScore()._single_turn_ascore(
            SingleTurnSample(user_input="dummy", response=response, reference=reference), None
        )
        v2_result = await BleuScore().ascore(reference=reference, response=response)

        assert legacy_score == pytest.approx(0.4755, abs=0.01)
        assert v2_result.value == pytest.approx(0.4755, abs=0.01)

    @pytest.mark.asyncio
    async def test_bleu_score_single_sentence_reference(self):
        reference = "The cat sat on the mat."
        response = "The cat sat on the mat."

        legacy_score = await LegacyBleuScore()._single_turn_ascore(
            SingleTurnSample(user_input="dummy", response=response, reference=reference), None
        )
        v2_result = await BleuScore().ascore(reference=reference, response=response)

        assert legacy_score == pytest.approx(1.0, abs=0.01)
        assert v2_result.value == pytest.approx(1.0, abs=0.01)  