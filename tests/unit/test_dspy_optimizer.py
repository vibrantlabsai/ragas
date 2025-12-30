from unittest.mock import MagicMock, Mock, patch

import pytest

from ragas.dataset_schema import SingleMetricAnnotation
from ragas.losses import MSELoss

try:
    import dspy  # noqa: F401

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


class TestDSPyOptimizer:
    @pytest.mark.skipif(DSPY_AVAILABLE, reason="dspy-ai is installed")
    def test_import_error_without_dspy(self):
        """Test that DSPyOptimizer raises ImportError when dspy-ai is not installed."""
        with pytest.raises(ImportError, match="DSPy optimizer requires dspy-ai"):
            from ragas.optimizers.dspy_optimizer import DSPyOptimizer

            DSPyOptimizer()

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_initialization_with_default_params(self):
        """Test DSPyOptimizer initialization with default parameters."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        optimizer = DSPyOptimizer()

        assert optimizer.num_candidates == 10
        assert optimizer.max_bootstrapped_demos == 5
        assert optimizer.max_labeled_demos == 5
        assert optimizer.init_temperature == 1.0
        assert optimizer._dspy is not None

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_initialization_with_custom_params(self):
        """Test DSPyOptimizer initialization with custom parameters."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        optimizer = DSPyOptimizer(
            num_candidates=20,
            max_bootstrapped_demos=10,
            max_labeled_demos=8,
            init_temperature=0.5,
        )

        assert optimizer.num_candidates == 20
        assert optimizer.max_bootstrapped_demos == 10
        assert optimizer.max_labeled_demos == 8
        assert optimizer.init_temperature == 0.5

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_initialization_with_all_params(self):
        """Test DSPyOptimizer initialization with all parameters."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        optimizer = DSPyOptimizer(
            num_candidates=15,
            max_bootstrapped_demos=7,
            max_labeled_demos=6,
            init_temperature=0.8,
            auto="heavy",
            num_threads=4,
            max_errors=5,
            seed=42,
            verbose=True,
            track_stats=False,
            log_dir="/tmp/dspy_logs",
            metric_threshold=0.9,
        )

        assert optimizer.num_candidates == 15
        assert optimizer.max_bootstrapped_demos == 7
        assert optimizer.max_labeled_demos == 6
        assert optimizer.init_temperature == 0.8
        assert optimizer.auto == "heavy"
        assert optimizer.num_threads == 4
        assert optimizer.max_errors == 5
        assert optimizer.seed == 42
        assert optimizer.verbose is True
        assert optimizer.track_stats is False
        assert optimizer.log_dir == "/tmp/dspy_logs"
        assert optimizer.metric_threshold == 0.9

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_validation_negative_num_candidates(self):
        """Test validation for negative num_candidates."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        with pytest.raises(ValueError, match="num_candidates must be positive"):
            DSPyOptimizer(num_candidates=-1)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_validation_negative_max_bootstrapped_demos(self):
        """Test validation for negative max_bootstrapped_demos."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        with pytest.raises(
            ValueError, match="max_bootstrapped_demos must be non-negative"
        ):
            DSPyOptimizer(max_bootstrapped_demos=-1)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_validation_negative_max_labeled_demos(self):
        """Test validation for negative max_labeled_demos."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        with pytest.raises(ValueError, match="max_labeled_demos must be non-negative"):
            DSPyOptimizer(max_labeled_demos=-1)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_validation_zero_init_temperature(self):
        """Test validation for zero init_temperature."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        with pytest.raises(ValueError, match="init_temperature must be positive"):
            DSPyOptimizer(init_temperature=0)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_validation_invalid_auto(self):
        """Test validation for invalid auto parameter."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        with pytest.raises(ValueError, match="auto must be"):
            DSPyOptimizer(auto="invalid")

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_validation_negative_num_threads(self):
        """Test validation for negative num_threads."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        with pytest.raises(ValueError, match="num_threads must be positive"):
            DSPyOptimizer(num_threads=-1)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_validation_negative_max_errors(self):
        """Test validation for negative max_errors."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        with pytest.raises(ValueError, match="max_errors must be non-negative"):
            DSPyOptimizer(max_errors=-1)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_validation_invalid_metric_threshold(self):
        """Test validation for metric_threshold out of range."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        with pytest.raises(
            ValueError, match="metric_threshold must be between 0 and 1"
        ):
            DSPyOptimizer(metric_threshold=1.5)

        with pytest.raises(
            ValueError, match="metric_threshold must be between 0 and 1"
        ):
            DSPyOptimizer(metric_threshold=-0.1)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_optimize_without_metric(self, fake_llm):
        """Test that optimize raises ValueError when no metric is set."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        optimizer = DSPyOptimizer()
        optimizer.llm = fake_llm

        dataset = Mock(spec=SingleMetricAnnotation)
        loss = MSELoss()

        with pytest.raises(ValueError, match="No metric provided"):
            optimizer.optimize(dataset, loss, {})

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_optimize_without_llm(self, fake_llm):
        """Test that optimize raises ValueError when no llm is set."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        optimizer = DSPyOptimizer()
        metric = Mock()
        optimizer.metric = metric

        dataset = Mock(spec=SingleMetricAnnotation)
        loss = MSELoss()

        with pytest.raises(ValueError, match="No llm provided"):
            optimizer.optimize(dataset, loss, {})

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    @patch("ragas.optimizers.dspy_adapter.setup_dspy_llm")
    @patch("ragas.optimizers.dspy_adapter.pydantic_prompt_to_dspy_signature")
    @patch("ragas.optimizers.dspy_adapter.ragas_dataset_to_dspy_examples")
    @patch("ragas.optimizers.dspy_adapter.create_dspy_metric")
    def test_optimize_basic_flow(
        self,
        mock_create_metric,
        mock_to_examples,
        mock_to_signature,
        mock_setup_llm,
        fake_llm,
    ):
        """Test basic optimization flow with mocked DSPy."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        optimizer = DSPyOptimizer()

        mock_metric = Mock()
        mock_metric.name = "test_metric"
        mock_metric.get_prompts.return_value = {
            "test_prompt": Mock(instruction="Test instruction")
        }
        optimizer.metric = mock_metric
        optimizer.llm = fake_llm

        mock_dspy = MagicMock()
        mock_signature = Mock()
        mock_to_signature.return_value = mock_signature

        mock_module = Mock()
        mock_dspy.Predict.return_value = mock_module

        mock_examples = [Mock()]
        mock_to_examples.return_value = mock_examples

        mock_metric_fn = Mock()
        mock_create_metric.return_value = mock_metric_fn

        mock_teleprompter = Mock()
        mock_optimized = Mock()
        mock_optimized.signature.instructions = "Optimized instruction"
        mock_teleprompter.compile.return_value = mock_optimized
        mock_dspy.MIPROv2.return_value = mock_teleprompter

        optimizer._dspy = mock_dspy

        dataset = Mock(spec=SingleMetricAnnotation)
        dataset.name = "test_metric"
        loss = MSELoss()

        result = optimizer.optimize(dataset, loss, {})

        assert "test_prompt" in result
        assert result["test_prompt"] == "Optimized instruction"

        mock_setup_llm.assert_called_once_with(mock_dspy, fake_llm)
        mock_metric.get_prompts.assert_called_once()
        mock_to_signature.assert_called_once()
        mock_to_examples.assert_called_once()
        mock_create_metric.assert_called_once_with(loss, "test_metric")

        mock_dspy.MIPROv2.assert_called_once_with(
            num_candidates=10,
            max_bootstrapped_demos=5,
            max_labeled_demos=5,
            init_temperature=1.0,
            auto="light",
            num_threads=None,
            max_errors=None,
            seed=9,
            verbose=False,
            track_stats=True,
            log_dir=None,
            metric_threshold=None,
        )

        mock_teleprompter.compile.assert_called_once_with(
            mock_module,
            trainset=mock_examples,
            metric=mock_metric_fn,
        )

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    @patch("ragas.optimizers.dspy_adapter.setup_dspy_llm")
    @patch("ragas.optimizers.dspy_adapter.pydantic_prompt_to_dspy_signature")
    @patch("ragas.optimizers.dspy_adapter.ragas_dataset_to_dspy_examples")
    @patch("ragas.optimizers.dspy_adapter.create_dspy_metric")
    def test_optimize_with_custom_params(
        self,
        mock_create_metric,
        mock_to_examples,
        mock_to_signature,
        mock_setup_llm,
        fake_llm,
    ):
        """Test that custom parameters are passed to MIPROv2."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        optimizer = DSPyOptimizer(
            num_candidates=15,
            max_bootstrapped_demos=7,
            max_labeled_demos=6,
            init_temperature=0.8,
            auto="heavy",
            num_threads=4,
            max_errors=5,
            seed=42,
            verbose=True,
            track_stats=False,
            log_dir="/tmp/dspy",
            metric_threshold=0.85,
        )

        mock_metric = Mock()
        mock_metric.name = "test_metric"
        mock_metric.get_prompts.return_value = {
            "test_prompt": Mock(instruction="Test instruction")
        }
        optimizer.metric = mock_metric
        optimizer.llm = fake_llm

        mock_dspy = MagicMock()
        mock_signature = Mock()
        mock_to_signature.return_value = mock_signature

        mock_module = Mock()
        mock_dspy.Predict.return_value = mock_module

        mock_examples = [Mock()]
        mock_to_examples.return_value = mock_examples

        mock_metric_fn = Mock()
        mock_create_metric.return_value = mock_metric_fn

        mock_teleprompter = Mock()
        mock_optimized = Mock()
        mock_optimized.signature.instructions = "Optimized instruction"
        mock_teleprompter.compile.return_value = mock_optimized
        mock_dspy.MIPROv2.return_value = mock_teleprompter

        optimizer._dspy = mock_dspy

        dataset = Mock(spec=SingleMetricAnnotation)
        dataset.name = "test_metric"
        loss = MSELoss()

        result = optimizer.optimize(dataset, loss, {})

        assert "test_prompt" in result

        mock_dspy.MIPROv2.assert_called_once_with(
            num_candidates=15,
            max_bootstrapped_demos=7,
            max_labeled_demos=6,
            init_temperature=0.8,
            auto="heavy",
            num_threads=4,
            max_errors=5,
            seed=42,
            verbose=True,
            track_stats=False,
            log_dir="/tmp/dspy",
            metric_threshold=0.85,
        )

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_extract_instruction_from_signature(self):
        """Test extracting instruction from optimized module with signature.instructions."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        optimizer = DSPyOptimizer()

        mock_module = Mock()
        mock_module.signature.instructions = "Test instruction"

        result = optimizer._extract_instruction(mock_module)
        assert result == "Test instruction"

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_extract_instruction_from_docstring(self):
        """Test extracting instruction from signature.__doc__."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        optimizer = DSPyOptimizer()

        mock_module = Mock()
        del mock_module.signature.instructions
        mock_module.signature.__doc__ = "Doc instruction"

        result = optimizer._extract_instruction(mock_module)
        assert result == "Doc instruction"

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_extract_instruction_from_extended_signature(self):
        """Test extracting instruction from extended_signature."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        optimizer = DSPyOptimizer()

        mock_module = Mock()
        del mock_module.signature
        mock_module.extended_signature = "Extended instruction"

        result = optimizer._extract_instruction(mock_module)
        assert result == "Extended instruction"

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_extract_instruction_fallback(self):
        """Test extracting instruction returns empty string as fallback."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        optimizer = DSPyOptimizer()

        mock_module = Mock(spec=[])

        result = optimizer._extract_instruction(mock_module)
        assert result == ""

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_cache_key_generation(self, fake_llm):
        """Test cache key generation is deterministic."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        optimizer = DSPyOptimizer()

        mock_metric = Mock()
        mock_metric.name = "test_metric"
        optimizer.metric = mock_metric
        optimizer.llm = fake_llm

        dataset = Mock(spec=SingleMetricAnnotation)
        dataset.model_dump.return_value = {"data": "test"}
        loss = MSELoss()
        config = {"test": "config"}

        key1 = optimizer._generate_cache_key(dataset, loss, config)
        key2 = optimizer._generate_cache_key(dataset, loss, config)

        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 64

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_cache_key_different_for_different_inputs(self, fake_llm):
        """Test cache key changes with different inputs."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        optimizer = DSPyOptimizer()

        mock_metric = Mock()
        mock_metric.name = "test_metric"
        optimizer.metric = mock_metric
        optimizer.llm = fake_llm

        dataset1 = Mock(spec=SingleMetricAnnotation)
        dataset1.model_dump.return_value = {"data": "test1"}
        dataset2 = Mock(spec=SingleMetricAnnotation)
        dataset2.model_dump.return_value = {"data": "test2"}

        loss = MSELoss()
        config = {"test": "config"}

        key1 = optimizer._generate_cache_key(dataset1, loss, config)
        key2 = optimizer._generate_cache_key(dataset2, loss, config)

        assert key1 != key2

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    @patch("ragas.optimizers.dspy_adapter.setup_dspy_llm")
    @patch("ragas.optimizers.dspy_adapter.pydantic_prompt_to_dspy_signature")
    @patch("ragas.optimizers.dspy_adapter.ragas_dataset_to_dspy_examples")
    @patch("ragas.optimizers.dspy_adapter.create_dspy_metric")
    def test_cache_hit(
        self,
        mock_create_metric,
        mock_to_examples,
        mock_to_signature,
        mock_setup_llm,
        fake_llm,
    ):
        """Test that cached results are returned on cache hit."""
        from ragas.cache import DiskCacheBackend
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        cache = DiskCacheBackend(cache_dir=".test_cache_dspy")
        optimizer = DSPyOptimizer(cache=cache)

        mock_metric = Mock()
        mock_metric.name = "test_metric"
        mock_metric.get_prompts.return_value = {
            "test_prompt": Mock(instruction="Test instruction")
        }
        optimizer.metric = mock_metric
        optimizer.llm = fake_llm

        mock_dspy = MagicMock()
        mock_signature = Mock()
        mock_to_signature.return_value = mock_signature

        mock_module = Mock()
        mock_dspy.Predict.return_value = mock_module

        mock_examples = [Mock()]
        mock_to_examples.return_value = mock_examples

        mock_metric_fn = Mock()
        mock_create_metric.return_value = mock_metric_fn

        mock_teleprompter = Mock()
        mock_optimized = Mock()
        mock_optimized.signature.instructions = "Optimized instruction"
        mock_teleprompter.compile.return_value = mock_optimized
        mock_dspy.MIPROv2.return_value = mock_teleprompter

        optimizer._dspy = mock_dspy

        dataset = Mock(spec=SingleMetricAnnotation)
        dataset.name = "test_metric"
        dataset.model_dump.return_value = {"data": "test"}
        loss = MSELoss()

        result1 = optimizer.optimize(dataset, loss, {})
        assert mock_teleprompter.compile.call_count == 1

        result2 = optimizer.optimize(dataset, loss, {})
        assert mock_teleprompter.compile.call_count == 1

        assert result1 == result2
        assert result1["test_prompt"] == "Optimized instruction"

        cache.cache.close()
        import shutil

        shutil.rmtree(".test_cache_dspy", ignore_errors=True)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    @patch("ragas.optimizers.dspy_adapter.setup_dspy_llm")
    @patch("ragas.optimizers.dspy_adapter.pydantic_prompt_to_dspy_signature")
    @patch("ragas.optimizers.dspy_adapter.ragas_dataset_to_dspy_examples")
    @patch("ragas.optimizers.dspy_adapter.create_dspy_metric")
    def test_cache_miss(
        self,
        mock_create_metric,
        mock_to_examples,
        mock_to_signature,
        mock_setup_llm,
        fake_llm,
    ):
        """Test that optimization runs on cache miss."""
        from ragas.cache import DiskCacheBackend
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        cache = DiskCacheBackend(cache_dir=".test_cache_dspy_miss")
        optimizer = DSPyOptimizer(cache=cache)

        mock_metric = Mock()
        mock_metric.name = "test_metric"
        mock_metric.get_prompts.return_value = {
            "test_prompt": Mock(instruction="Test instruction")
        }
        optimizer.metric = mock_metric
        optimizer.llm = fake_llm

        mock_dspy = MagicMock()
        mock_signature = Mock()
        mock_to_signature.return_value = mock_signature

        mock_module = Mock()
        mock_dspy.Predict.return_value = mock_module

        mock_examples = [Mock()]
        mock_to_examples.return_value = mock_examples

        mock_metric_fn = Mock()
        mock_create_metric.return_value = mock_metric_fn

        mock_teleprompter = Mock()
        mock_optimized = Mock()
        mock_optimized.signature.instructions = "Optimized instruction"
        mock_teleprompter.compile.return_value = mock_optimized
        mock_dspy.MIPROv2.return_value = mock_teleprompter

        optimizer._dspy = mock_dspy

        dataset1 = Mock(spec=SingleMetricAnnotation)
        dataset1.name = "test_metric"
        dataset1.model_dump.return_value = {"data": "test1"}

        dataset2 = Mock(spec=SingleMetricAnnotation)
        dataset2.name = "test_metric"
        dataset2.model_dump.return_value = {"data": "test2"}

        loss = MSELoss()

        result1 = optimizer.optimize(dataset1, loss, {})
        assert mock_teleprompter.compile.call_count == 1

        result2 = optimizer.optimize(dataset2, loss, {})
        assert mock_teleprompter.compile.call_count == 2

        assert result1["test_prompt"] == "Optimized instruction"
        assert result2["test_prompt"] == "Optimized instruction"

        cache.cache.close()
        import shutil

        shutil.rmtree(".test_cache_dspy_miss", ignore_errors=True)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
    def test_optimize_without_cache(self, fake_llm):
        """Test that optimization works without cache configured."""
        from ragas.optimizers.dspy_optimizer import DSPyOptimizer

        optimizer = DSPyOptimizer(cache=None)

        assert optimizer.cache is None
