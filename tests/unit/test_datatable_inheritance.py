"""Tests for DataTable inheritance and return type correctness."""

import tempfile
import typing as t

import pytest
from pydantic import BaseModel

from ragas import Experiment
from ragas.backends.local_csv import LocalCSVBackend
from ragas.dataset import Dataset, DataTable


# Test BaseModel classes
class SimpleTestModel(BaseModel):
    name: str
    age: int
    score: float


class ComplexTestModel(BaseModel):
    id: int
    metadata: t.Dict[str, t.Any]
    tags: t.List[str]


# Test fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def mock_backend(temp_dir):
    """Create a mock backend for testing."""
    return LocalCSVBackend(temp_dir)


@pytest.fixture
def simple_test_data():
    """Simple test data for testing."""
    return [
        {"name": "Alice", "age": 30, "score": 85.5},
        {"name": "Bob", "age": 25, "score": 92.0},
        {"name": "Charlie", "age": 35, "score": 78.5},
    ]


@pytest.fixture
def complex_test_data():
    """Complex test data for testing."""
    return [
        {
            "id": 1,
            "metadata": {"score": 0.85, "tags": ["test", "important"]},
            "tags": ["evaluation", "metrics"],
        },
        {
            "id": 2,
            "metadata": {"score": 0.92, "tags": ["production"]},
            "tags": ["benchmark", "validation"],
        },
    ]


class TestDataTableInheritance:
    """Test that DataTable subclasses preserve their type in method returns."""

    def test_dataset_load_returns_dataset(self, mock_backend, simple_test_data):
        """Test that Dataset.load() returns a Dataset instance, not DataTable."""
        # Save data first
        mock_backend.save_dataset("test_dataset", simple_test_data)

        # Load using Dataset.load()
        result = Dataset.load("test_dataset", mock_backend)

        # This should be a Dataset instance, not just DataTable
        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert not isinstance(result, DataTable) or isinstance(result, Dataset), (
            "Dataset.load() should return Dataset, not DataTable"
        )

    def test_dataset_load_with_model_returns_dataset(
        self, mock_backend, simple_test_data
    ):
        """Test that Dataset.load() with model returns a Dataset instance."""
        # Save data first
        mock_backend.save_dataset("test_dataset", simple_test_data)

        # Load using Dataset.load() with model
        result = Dataset.load("test_dataset", mock_backend, SimpleTestModel)

        # This should be a Dataset instance
        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert result.data_model == SimpleTestModel

    def test_dataset_validate_with_returns_dataset(
        self, mock_backend, simple_test_data
    ):
        """Test that Dataset.validate_with() returns a Dataset instance."""
        # Create unvalidated dataset
        dataset = Dataset("test_dataset", mock_backend, data=simple_test_data)

        # Validate with model
        result = dataset.validate_with(SimpleTestModel)

        # This should be a Dataset instance, not just DataTable
        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert result.data_model == SimpleTestModel

    def test_experiment_load_returns_experiment(self, mock_backend, simple_test_data):
        """Test that Experiment.load() returns an Experiment instance."""
        # Save data first
        mock_backend.save_experiment("test_experiment", simple_test_data)

        # Load using Experiment.load()
        result = Experiment.load("test_experiment", mock_backend)

        # This should be an Experiment instance, not just DataTable
        assert isinstance(result, Experiment), (
            f"Expected Experiment, got {type(result)}"
        )

    def test_experiment_load_with_model_returns_experiment(
        self, mock_backend, simple_test_data
    ):
        """Test that Experiment.load() with model returns an Experiment instance."""
        # Save data first
        mock_backend.save_experiment("test_experiment", simple_test_data)

        # Load using Experiment.load() with model
        result = Experiment.load("test_experiment", mock_backend, SimpleTestModel)

        # This should be an Experiment instance
        assert isinstance(result, Experiment), (
            f"Expected Experiment, got {type(result)}"
        )
        assert result.data_model == SimpleTestModel

    def test_experiment_validate_with_returns_experiment(
        self, mock_backend, simple_test_data
    ):
        """Test that Experiment.validate_with() returns an Experiment instance."""
        # Create unvalidated experiment
        experiment = Experiment("test_experiment", mock_backend, data=simple_test_data)

        # Validate with model
        result = experiment.validate_with(SimpleTestModel)

        # This should be an Experiment instance, not just DataTable
        assert isinstance(result, Experiment), (
            f"Expected Experiment, got {type(result)}"
        )
        assert result.data_model == SimpleTestModel


class TestDatasetMethods:
    """Test Dataset-specific behavior."""

    def test_dataset_type_preservation_through_operations(
        self, mock_backend, simple_test_data
    ):
        """Test that Dataset type is preserved through multiple operations."""
        # Save data first
        mock_backend.save_dataset("test_dataset", simple_test_data)

        # Load -> validate -> should still be Dataset
        loaded = Dataset.load("test_dataset", mock_backend)
        validated = loaded.validate_with(SimpleTestModel)

        assert isinstance(loaded, Dataset)
        assert isinstance(validated, Dataset)
        assert validated.data_model == SimpleTestModel

    def test_dataset_str_representation(self, mock_backend, simple_test_data):
        """Test that Dataset shows correct type in string representation."""
        dataset = Dataset("test_dataset", mock_backend, data=simple_test_data)
        str_repr = str(dataset)

        # Should show "Dataset" not "DataTable"
        assert "Dataset" in str_repr
        assert "DataTable" not in str_repr or "Dataset" in str_repr


class TestExperimentMethods:
    """Test Experiment-specific behavior."""

    def test_experiment_type_preservation_through_operations(
        self, mock_backend, simple_test_data
    ):
        """Test that Experiment type is preserved through multiple operations."""
        # Save data first
        mock_backend.save_experiment("test_experiment", simple_test_data)

        # Load -> validate -> should still be Experiment
        loaded = Experiment.load("test_experiment", mock_backend)
        validated = loaded.validate_with(SimpleTestModel)

        assert isinstance(loaded, Experiment)
        assert isinstance(validated, Experiment)
        assert validated.data_model == SimpleTestModel

    def test_experiment_str_representation(self, mock_backend, simple_test_data):
        """Test that Experiment shows correct type in string representation."""
        experiment = Experiment("test_experiment", mock_backend, data=simple_test_data)
        str_repr = str(experiment)

        # Should show "Experiment" not "DataTable"
        assert "Experiment" in str_repr
        assert "DataTable" not in str_repr or "Experiment" in str_repr


class TestTypeAnnotations:
    """Test that type annotations are correct for static type checking."""

    def test_dataset_load_type_annotation(self, mock_backend, simple_test_data):
        """Test that Dataset.load() has correct type annotation."""
        # Save data first
        mock_backend.save_dataset("test_dataset", simple_test_data)

        # This should type-check correctly
        result: Dataset = Dataset.load("test_dataset", mock_backend)
        assert isinstance(result, Dataset)

    def test_dataset_validate_with_type_annotation(
        self, mock_backend, simple_test_data
    ):
        """Test that Dataset.validate_with() has correct type annotation."""
        dataset = Dataset("test_dataset", mock_backend, data=simple_test_data)

        # This should type-check correctly
        result: Dataset = dataset.validate_with(SimpleTestModel)
        assert isinstance(result, Dataset)

    def test_experiment_load_type_annotation(self, mock_backend, simple_test_data):
        """Test that Experiment.load() has correct type annotation."""
        # Save data first
        mock_backend.save_experiment("test_experiment", simple_test_data)

        # This should type-check correctly
        result: Experiment = Experiment.load("test_experiment", mock_backend)
        assert isinstance(result, Experiment)

    def test_experiment_validate_with_type_annotation(
        self, mock_backend, simple_test_data
    ):
        """Test that Experiment.validate_with() has correct type annotation."""
        experiment = Experiment("test_experiment", mock_backend, data=simple_test_data)

        # This should type-check correctly
        result: Experiment = experiment.validate_with(SimpleTestModel)
        assert isinstance(result, Experiment)


class TestComplexDataHandling:
    """Test that inheritance works correctly with complex data."""

    def test_dataset_complex_data_preservation(self, mock_backend, complex_test_data):
        """Test Dataset with complex data maintains type."""
        dataset = Dataset("test_dataset", mock_backend, data=complex_test_data)

        # Validate should return Dataset
        validated = dataset.validate_with(ComplexTestModel)
        assert isinstance(validated, Dataset)

    def test_experiment_complex_data_preservation(
        self, mock_backend, complex_test_data
    ):
        """Test Experiment with complex data maintains type."""
        experiment = Experiment("test_experiment", mock_backend, data=complex_test_data)

        # Validate should return Experiment
        validated = experiment.validate_with(ComplexTestModel)
        assert isinstance(validated, Experiment)
