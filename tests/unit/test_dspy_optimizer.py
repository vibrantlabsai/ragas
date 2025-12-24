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
        )

        mock_teleprompter.compile.assert_called_once_with(
            mock_module,
            trainset=mock_examples,
            metric=mock_metric_fn,
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
