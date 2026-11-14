"""Comprehensive tests for LocalCSVBackend to test serialization edge cases."""

import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, ValidationError

from ragas.backends.local_csv import LocalCSVBackend


# Test BaseModel classes
class SimpleTestModel(BaseModel):
    name: str
    age: int
    score: float
    is_active: bool


class ComplexTestModel(BaseModel):
    id: int
    metadata: Dict[str, Any]
    tags: List[str]
    config: Optional[Dict[str, Any]] = None
    created_at: datetime


class NestedTestModel(BaseModel):
    user: SimpleTestModel
    settings: Dict[str, Any]
    history: List[Dict[str, Any]]


# Test fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def backend(temp_dir):
    """Create a LocalCSVBackend instance with temp directory."""
    return LocalCSVBackend(temp_dir)


@pytest.fixture
def simple_data():
    """Simple test data with basic types."""
    return [
        {"name": "Alice", "age": 30, "score": 85.5, "is_active": True},
        {"name": "Bob", "age": 25, "score": 92.0, "is_active": False},
        {"name": "Charlie", "age": 35, "score": 78.5, "is_active": True},
    ]


@pytest.fixture
def complex_data():
    """Complex test data with nested structures."""
    return [
        {
            "id": 1,
            "metadata": {"score": 0.85, "tags": ["test", "important"]},
            "tags": ["evaluation", "metrics"],
            "config": {"model": "gpt-4", "temperature": 0.7},
            "created_at": datetime(2024, 1, 15, 10, 30, 0),
        },
        {
            "id": 2,
            "metadata": {"score": 0.92, "tags": ["production"]},
            "tags": ["benchmark", "validation"],
            "config": {"model": "claude-3", "temperature": 0.5},
            "created_at": datetime(2024, 1, 16, 14, 45, 0),
        },
    ]


@pytest.fixture
def nested_data():
    """Deeply nested test data."""
    return [
        {
            "user": {"name": "Alice", "age": 30, "score": 85.5, "is_active": True},
            "settings": {
                "theme": "dark",
                "notifications": {"email": True, "push": False},
                "features": ["advanced", "beta"],
            },
            "history": [
                {"action": "login", "timestamp": "2024-01-15T10:30:00"},
                {"action": "query", "timestamp": "2024-01-15T10:35:00"},
            ],
        }
    ]


# 1. Basic Functionality Tests
class TestBasicFunctionality:
    """Test basic LocalCSVBackend functionality."""

    def test_initialization(self, temp_dir):
        """Test backend initialization."""
        backend = LocalCSVBackend(temp_dir)
        assert backend.root_dir == Path(temp_dir)

    def test_get_data_dir(self, backend):
        """Test data directory path generation."""
        datasets_dir = backend._get_data_dir("datasets")
        experiments_dir = backend._get_data_dir("experiments")

        assert datasets_dir.name == "datasets"
        assert experiments_dir.name == "experiments"

    def test_get_file_path(self, backend):
        """Test file path generation."""
        dataset_path = backend._get_file_path("datasets", "test_dataset")
        experiment_path = backend._get_file_path("experiments", "test_experiment")

        assert dataset_path.name == "test_dataset.csv"
        assert experiment_path.name == "test_experiment.csv"

    def test_save_and_load_simple_data(self, backend, simple_data):
        """Test basic save and load cycle with simple data."""
        # Save dataset
        backend.save_dataset("test_simple", simple_data)

        # Load dataset
        loaded_data = backend.load_dataset("test_simple")

        # Verify data structure (note: all values become strings in CSV)
        assert len(loaded_data) == len(simple_data)
        assert loaded_data[0]["name"] == "Alice"
        # This will fail because CSV converts everything to strings
        # assert loaded_data[0]["age"] == 30  # This will be "30"

    def test_directory_creation(self, backend, simple_data):
        """Test automatic directory creation."""
        # Directories shouldn't exist initially
        datasets_dir = backend._get_data_dir("datasets")
        experiments_dir = backend._get_data_dir("experiments")
        assert not datasets_dir.exists()
        assert not experiments_dir.exists()

        # Save data should create directories
        backend.save_dataset("test", simple_data)
        backend.save_experiment("test", simple_data)

        # Directories should now exist
        assert datasets_dir.exists()
        assert experiments_dir.exists()

    def test_list_datasets_and_experiments(self, backend, simple_data):
        """Test listing datasets and experiments."""
        # Initially empty
        assert backend.list_datasets() == []
        assert backend.list_experiments() == []

        # Save some data
        backend.save_dataset("dataset1", simple_data)
        backend.save_dataset("dataset2", simple_data)
        backend.save_experiment("experiment1", simple_data)

        # Check listings
        datasets = backend.list_datasets()
        experiments = backend.list_experiments()

        assert sorted(datasets) == ["dataset1", "dataset2"]
        assert experiments == ["experiment1"]

    def test_save_empty_data(self, backend):
        """Test saving empty datasets."""
        backend.save_dataset("empty_dataset", [])

        # Should create empty file
        file_path = backend._get_file_path("datasets", "empty_dataset")
        assert file_path.exists()

        # Loading should return empty list
        loaded_data = backend.load_dataset("empty_dataset")
        assert loaded_data == []


# 2. Data Type Edge Cases (The Real Challenge)
class TestDataTypeEdgeCases:
    """Test complex data types that reveal CSV serialization issues."""

    @pytest.mark.skip(reason="CSV backend doesn't support nested dictionaries")
    def test_nested_dictionaries(self, backend):
        """Test nested dictionary serialization - THIS SHOULD FAIL."""
        data = [
            {
                "id": 1,
                "metadata": {"score": 0.85, "tags": ["test", "important"]},
                "config": {"model": "gpt-4", "settings": {"temperature": 0.7}},
            }
        ]

        backend.save_dataset("nested_test", data)
        loaded_data = backend.load_dataset("nested_test")

        # This will fail - nested dicts become string representations
        assert loaded_data[0]["metadata"] == {
            "score": 0.85,
            "tags": ["test", "important"],
        }

        # Show what actually happens
        print(f"Original: {data[0]['metadata']}")
        print(f"Loaded: {loaded_data[0]['metadata']}")
        print(f"Type: {type(loaded_data[0]['metadata'])}")

    @pytest.mark.skip(reason="CSV backend doesn't support lists of objects")
    def test_lists_of_objects(self, backend):
        """Test lists of objects serialization - THIS SHOULD FAIL."""
        data = [
            {
                "id": 1,
                "results": [
                    {"metric": "accuracy", "value": 0.9},
                    {"metric": "precision", "value": 0.8},
                ],
            }
        ]

        backend.save_dataset("list_test", data)
        loaded_data = backend.load_dataset("list_test")

        # This will fail - lists become string representations
        assert loaded_data[0]["results"][0]["metric"] == "accuracy"

        # Show what actually happens
        print(f"Original: {data[0]['results']}")
        print(f"Loaded: {loaded_data[0]['results']}")
        print(f"Type: {type(loaded_data[0]['results'])}")

    @pytest.mark.skip(reason="CSV backend doesn't preserve data types")
    def test_mixed_types(self, backend):
        """Test mixed data types - THIS WILL PARTIALLY FAIL."""
        data = [
            {
                "str_field": "text",
                "int_field": 42,
                "float_field": 3.14,
                "bool_field": True,
                "null_field": None,
            }
        ]

        backend.save_dataset("mixed_test", data)
        loaded_data = backend.load_dataset("mixed_test")

        # All values become strings in CSV - these assertions should fail
        assert loaded_data[0]["str_field"] == "text"  # This works
        assert loaded_data[0]["int_field"] == 42  # This will fail - it's "42" not 42
        assert (
            loaded_data[0]["float_field"] == 3.14
        )  # This will fail - it's "3.14" not 3.14
        assert (
            loaded_data[0]["bool_field"] is True
        )  # This will fail - it's "True" not True

    @pytest.mark.skip(reason="CSV backend doesn't support datetime objects")
    def test_datetime_objects(self, backend):
        """Test datetime serialization - THIS SHOULD FAIL."""
        data = [
            {
                "id": 1,
                "created_at": datetime(2024, 1, 15, 10, 30, 0),
                "updated_date": date(2024, 1, 16),
            }
        ]

        backend.save_dataset("datetime_test", data)
        loaded_data = backend.load_dataset("datetime_test")

        # Datetime objects become string representations - this should fail
        original_dt = data[0]["created_at"]
        loaded_dt = loaded_data[0]["created_at"]

        assert isinstance(original_dt, datetime)
        assert isinstance(loaded_dt, datetime)  # This will fail - it's a string now!

    @pytest.mark.skip(reason="CSV backend doesn't support complex nested structures")
    def test_complex_nested_structure(self, backend):
        """Test deeply nested structures - THIS SHOULD FAIL BADLY."""
        data = [
            {
                "config": {
                    "database": {
                        "host": "localhost",
                        "ports": [5432, 5433],
                        "credentials": {"user": "admin", "encrypted": True},
                    },
                    "features": ["auth", "logging"],
                }
            }
        ]

        backend.save_dataset("complex_test", data)
        loaded_data = backend.load_dataset("complex_test")

        # This will fail - complex nested structure becomes string
        assert loaded_data[0]["config"]["database"]["host"] == "localhost"

        # Show the mangled data
        print(f"Original: {data[0]['config']}")
        print(f"Loaded: {loaded_data[0]['config']}")


# 3. BaseModel Integration Tests
class TestBaseModelIntegration:
    """Test BaseModel validation and conversion."""

    def test_simple_basemodel_save_load(self, backend, simple_data):
        """Test BaseModel with simple data types."""
        # Save raw data
        backend.save_dataset("simple_model_test", simple_data, SimpleTestModel)

        # Load and validate with BaseModel
        loaded_data = backend.load_dataset("simple_model_test")

        # Try to create BaseModel instances - this will partially fail
        try:
            models = [SimpleTestModel(**item) for item in loaded_data]
            print("BaseModel creation succeeded!")
            print(f"First model: {models[0]}")
        except Exception as e:
            print(f"BaseModel creation failed: {e}")
            print(
                f"Loaded data types: {[(k, type(v)) for k, v in loaded_data[0].items()]}"
            )

    @pytest.mark.skip(reason="CSV backend doesn't support complex BaseModel validation")
    def test_complex_basemodel_roundtrip(self, backend, complex_data):
        """Test BaseModel with complex data - THIS SHOULD FAIL."""
        # Save raw data
        backend.save_dataset("complex_model_test", complex_data, ComplexTestModel)

        # Load and try to validate
        loaded_data = backend.load_dataset("complex_model_test")

        # This will fail because nested structures are corrupted
        with pytest.raises(ValidationError):
            [ComplexTestModel(**item) for item in loaded_data]

    def test_basemodel_type_coercion(self, backend):
        """Test BaseModel's ability to coerce string types."""
        # Data that should be coercible from strings
        data = [{"name": "Alice", "age": "30", "score": "85.5", "is_active": "true"}]

        backend.save_dataset("coercion_test", data)
        loaded_data = backend.load_dataset("coercion_test")

        # Pydantic should be able to handle some string-to-type conversions
        # This might work for simple types
        model = SimpleTestModel(**loaded_data[0])
        print(f"Type coercion successful: {model}")
        assert model.age == 30  # String "30" -> int 30
        assert model.score == 85.5  # String "85.5" -> float 85.5


# 4. Error Handling & Edge Cases
class TestErrorHandling:
    """Test error scenarios and edge cases."""

    def test_load_nonexistent_file(self, backend):
        """Test loading non-existent files."""
        with pytest.raises(FileNotFoundError):
            backend.load_dataset("nonexistent")

        with pytest.raises(FileNotFoundError):
            backend.load_experiment("nonexistent")

    def test_unicode_and_special_characters(self, backend):
        """Test handling of unicode and special characters."""
        data = [
            {
                "name": "José María",
                "description": "Testing émojis 🚀 and spëcial chars",
                "chinese": "你好世界",
                "symbols": "!@#$%^&*()_+{}[]|;:,.<>?",
            }
        ]

        backend.save_dataset("unicode_test", data)
        loaded_data = backend.load_dataset("unicode_test")

        # Unicode should be preserved
        assert loaded_data[0]["name"] == "José María"
        assert loaded_data[0]["chinese"] == "你好世界"
        assert "🚀" in loaded_data[0]["description"]

    def test_csv_injection_protection(self, backend):
        """Test protection against CSV injection attacks."""
        # CSV injection attempts
        data = [
            {
                "formula": "=SUM(A1:A10)",
                "command": "@SUM(A1:A10)",
                "plus_formula": "+SUM(A1:A10)",
                "minus_formula": "-SUM(A1:A10)",
            }
        ]

        backend.save_dataset("injection_test", data)
        loaded_data = backend.load_dataset("injection_test")

        # Data should be preserved as-is (strings)
        assert loaded_data[0]["formula"] == "=SUM(A1:A10)"

    def test_empty_and_null_values(self, backend):
        """Test handling of empty and null values."""
        data = [
            {
                "empty_string": "",
                "null_value": None,
                "whitespace": "   ",
                "zero": 0,
                "false": False,
            }
        ]

        backend.save_dataset("empty_test", data)
        loaded_data = backend.load_dataset("empty_test")

        # Show how null values are handled
        print(f"Original null: {data[0]['null_value']}")
        print(f"Loaded null: {loaded_data[0]['null_value']}")
        print(f"Loaded empty: '{loaded_data[0]['empty_string']}'")

    def test_large_text_fields(self, backend):
        """Test handling of large text fields."""
        large_text = "A" * 10000  # 10KB of text
        data = [
            {
                "id": 1,
                "large_field": large_text,
                "normal_field": "small",
            }
        ]

        backend.save_dataset("large_text_test", data)
        loaded_data = backend.load_dataset("large_text_test")

        # Large text should be preserved
        assert len(loaded_data[0]["large_field"]) == 10000
        assert loaded_data[0]["large_field"] == large_text

    def test_heterogeneous_row_keys(self, backend):
        """Rows with different key sets must not crash save_dataset.

        Dataset.append()/Experiment rows aren't required to share identical keys
        when no data_model/validation is configured (e.g. only some rows carry an
        optional "reason" field). csv.DictWriter defaults to extrasaction="raise",
        so if fieldnames are taken from only the first row, any later row with an
        extra key raises ValueError instead of being saved.
        """
        data = [
            {"question": "capital of France?", "response": "Paris", "score": 1.0},
            {
                "question": "capital of UK?",
                "response": "London",
                "score": 0.0,
                "reason": "wrong span matched",
            },
        ]

        backend.save_dataset("heterogeneous_test", data)
        loaded_data = backend.load_dataset("heterogeneous_test")

        assert len(loaded_data) == 2
        # Row that never had "reason" should round-trip as an empty string, not
        # be dropped or misaligned.
        assert loaded_data[0]["reason"] == ""
        assert loaded_data[1]["reason"] == "wrong span matched"

    def test_malformed_csv_handling(self, backend, temp_dir):
        """Test behavior with malformed CSV files."""
        # Create a malformed CSV file manually
        malformed_csv = Path(temp_dir) / "datasets" / "malformed.csv"
        malformed_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(malformed_csv, "w") as f:
            f.write("header1,header2\n")
            f.write("value1,value2,extra_value\n")  # Too many columns
            f.write("value3\n")  # Too few columns

        # Try to load malformed CSV
        try:
            loaded_data = backend.load_dataset("malformed")
            print(f"Malformed CSV loaded: {loaded_data}")
        except Exception as e:
            print(f"Malformed CSV failed to load: {e}")
