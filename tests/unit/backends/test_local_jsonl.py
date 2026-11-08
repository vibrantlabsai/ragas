"""Comprehensive tests for LocalJSONLBackend to test serialization capabilities."""

import tempfile
import typing as t
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

from ragas.backends.local_jsonl import LocalJSONLBackend


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


@pytest.fixture(name="backend")
def jsonl_backend_fixture(temp_dir):
    """Create a LocalJSONLBackend instance with temp directory."""
    return LocalJSONLBackend(temp_dir)


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
    """Test basic LocalJSONLBackend functionality."""

    def test_initialization(self, temp_dir):
        """Test backend initialization."""
        backend = LocalJSONLBackend(temp_dir)
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

        assert dataset_path.name == "test_dataset.jsonl"
        assert experiment_path.name == "test_experiment.jsonl"

    def test_save_and_load_simple_data(self, backend, simple_data):
        """Test basic save and load cycle with simple data."""
        # Save dataset
        backend.save_dataset("test_simple", simple_data)

        # Load dataset
        loaded_data = backend.load_dataset("test_simple")

        # Verify data structure - JSONL should preserve types
        assert len(loaded_data) == len(simple_data)
        assert loaded_data[0]["name"] == "Alice"
        assert loaded_data[0]["age"] == 30  # Should be int, not string
        assert loaded_data[0]["score"] == 85.5  # Should be float, not string
        assert loaded_data[0]["is_active"] is True  # Should be bool, not string

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
    """Test complex data types that JSONL should handle properly."""

    def test_nested_dictionaries(self, backend):
        """Test nested dictionary serialization - JSONL should handle this."""
        data = [
            {
                "id": 1,
                "metadata": {"score": 0.85, "tags": ["test", "important"]},
                "config": {"model": "gpt-4", "settings": {"temperature": 0.7}},
            }
        ]

        backend.save_dataset("nested_test", data)
        loaded_data = backend.load_dataset("nested_test")

        # JSONL should preserve nested dictionaries exactly
        assert loaded_data[0]["metadata"] == {
            "score": 0.85,
            "tags": ["test", "important"],
        }
        assert loaded_data[0]["config"]["settings"]["temperature"] == 0.7

    def test_lists_of_objects(self, backend):
        """Test lists of objects serialization - JSONL should handle this."""
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

        # JSONL should preserve lists of objects
        assert loaded_data[0]["results"][0]["metric"] == "accuracy"
        assert loaded_data[0]["results"][0]["value"] == 0.9
        assert loaded_data[0]["results"][1]["metric"] == "precision"
        assert loaded_data[0]["results"][1]["value"] == 0.8

    def test_mixed_types(self, backend):
        """Test mixed data types - JSONL should preserve all types."""
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

        # JSONL should preserve all data types
        assert loaded_data[0]["str_field"] == "text"
        assert loaded_data[0]["int_field"] == 42  # Should be int
        assert loaded_data[0]["float_field"] == 3.14  # Should be float
        assert loaded_data[0]["bool_field"] is True  # Should be bool
        assert loaded_data[0]["null_field"] is None  # Should be None

    def test_datetime_objects(self, backend):
        """Test datetime serialization - JSONL should handle this with ISO format."""
        data = [
            {
                "id": 1,
                "created_at": datetime(2024, 1, 15, 10, 30, 0),
                "updated_date": date(2024, 1, 16),
            }
        ]

        backend.save_dataset("datetime_test", data)
        loaded_data = backend.load_dataset("datetime_test")

        # JSONL should either preserve datetime objects or convert to ISO strings
        # For now, let's expect ISO strings that can be parsed back
        original_dt = data[0]["created_at"]
        loaded_dt = loaded_data[0]["created_at"]

        # Should be either datetime object or ISO string
        assert isinstance(original_dt, datetime)
        if isinstance(loaded_dt, str):
            # If string, should be valid ISO format
            parsed_dt = datetime.fromisoformat(loaded_dt.replace("Z", "+00:00"))
            assert parsed_dt.year == 2024
            assert parsed_dt.month == 1
            assert parsed_dt.day == 15
        else:
            # If datetime object, should be exact match
            assert loaded_dt == original_dt

    def test_complex_nested_structure(self, backend):
        """Test deeply nested structures - JSONL should handle this perfectly."""
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

        # JSONL should preserve complex nested structures exactly
        assert loaded_data[0]["config"]["database"]["host"] == "localhost"
        assert loaded_data[0]["config"]["database"]["ports"] == [5432, 5433]
        assert loaded_data[0]["config"]["database"]["credentials"]["user"] == "admin"
        assert loaded_data[0]["config"]["database"]["credentials"]["encrypted"] is True
        assert loaded_data[0]["config"]["features"] == ["auth", "logging"]


# 3. BaseModel Integration Tests
class TestBaseModelIntegration:
    """Test BaseModel validation and conversion."""

    def test_simple_basemodel_save_load(self, backend, simple_data):
        """Test BaseModel with simple data types."""
        # Save raw data
        backend.save_dataset("simple_model_test", simple_data, SimpleTestModel)

        # Load and validate with BaseModel
        loaded_data = backend.load_dataset("simple_model_test")

        # JSONL should enable perfect BaseModel roundtrip
        models = [SimpleTestModel(**item) for item in loaded_data]
        assert len(models) == 3
        assert models[0].name == "Alice"
        assert models[0].age == 30
        assert models[0].score == 85.5
        assert models[0].is_active is True

    def test_complex_basemodel_roundtrip(self, backend, complex_data):
        """Test BaseModel with complex data - JSONL should handle this."""
        # Save raw data
        backend.save_dataset("complex_model_test", complex_data, ComplexTestModel)

        # Load and try to validate
        loaded_data = backend.load_dataset("complex_model_test")

        # JSONL should enable perfect BaseModel validation
        models = [ComplexTestModel(**item) for item in loaded_data]
        assert len(models) == 2
        assert models[0].id == 1
        assert models[0].metadata["score"] == 0.85
        assert models[0].tags == ["evaluation", "metrics"]
        assert models[0].config is not None and models[0].config["model"] == "gpt-4"

    def test_basemodel_type_coercion(self, backend):
        """Test BaseModel's ability to coerce string types."""
        # Data that should be coercible from strings
        data = [{"name": "Alice", "age": "30", "score": "85.5", "is_active": "true"}]

        backend.save_dataset("coercion_test", data)
        loaded_data = backend.load_dataset("coercion_test")

        # JSONL + Pydantic should handle type coercion perfectly
        model = SimpleTestModel(**loaded_data[0])
        assert model.name == "Alice"
        assert model.age == 30  # String "30" -> int 30
        assert model.score == 85.5  # String "85.5" -> float 85.5
        # Note: "true" -> bool True coercion depends on implementation


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

        # Unicode should be preserved perfectly in JSONL
        assert loaded_data[0]["name"] == "José María"
        assert loaded_data[0]["chinese"] == "你好世界"
        assert "🚀" in loaded_data[0]["description"]

    def test_json_special_characters(self, backend):
        """Test handling of JSON special characters."""
        data = [
            {
                "quotes": 'He said "Hello World"',
                "backslashes": "C:\\Users\\test\\file.txt",
                "newlines": "Line 1\nLine 2\nLine 3",
                "tabs": "Column1\tColumn2\tColumn3",
            }
        ]

        backend.save_dataset("special_chars_test", data)
        loaded_data = backend.load_dataset("special_chars_test")

        # JSONL should handle JSON special characters properly
        assert loaded_data[0]["quotes"] == 'He said "Hello World"'
        assert loaded_data[0]["backslashes"] == "C:\\Users\\test\\file.txt"
        assert loaded_data[0]["newlines"] == "Line 1\nLine 2\nLine 3"
        assert loaded_data[0]["tabs"] == "Column1\tColumn2\tColumn3"

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

        # JSONL should handle null values properly
        assert loaded_data[0]["empty_string"] == ""
        assert loaded_data[0]["null_value"] is None
        assert loaded_data[0]["whitespace"] == "   "
        assert loaded_data[0]["zero"] == 0
        assert loaded_data[0]["false"] is False

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

        # Large text should be preserved perfectly
        assert len(loaded_data[0]["large_field"]) == 10000
        assert loaded_data[0]["large_field"] == large_text

    def test_malformed_jsonl_handling(self, backend, temp_dir):
        """Test behavior with malformed JSONL files."""
        # Create a malformed JSONL file manually
        malformed_jsonl = Path(temp_dir) / "datasets" / "malformed.jsonl"
        malformed_jsonl.parent.mkdir(parents=True, exist_ok=True)

        with open(malformed_jsonl, "w") as f:
            f.write('{"valid": "json"}\n')
            f.write('{"invalid": json}\n')  # Invalid JSON
            f.write('{"another": "valid"}\n')

        # Try to load malformed JSONL
        try:
            loaded_data = backend.load_dataset("malformed")
            # Should either handle gracefully or raise appropriate error
            print(f"Malformed JSONL loaded: {loaded_data}")
        except Exception as e:
            print(f"Malformed JSONL failed to load: {e}")
            # This is acceptable behavior


# Helper functions for debugging
def print_jsonl_content(jsonl_backend, data_type, name):
    """Helper to print raw JSONL content for debugging."""
    file_path = backend._get_file_path(data_type, name)
    if file_path.exists():
        print(f"\n=== JSONL Content for {name} ===")
        with open(file_path, "r") as f:
            print(f.read())
        print("=== End JSONL Content ===\n")


class TestDateLikeStringRoundTrip:
    """Strings that look like ISO dates must round trip as strings (#2865)."""

    def test_save_load_preserves_date_like_strings(self, backend):
        """A saved string that looks like a date / datetime must stay a string."""
        data = [
            {
                "id": "2026-07-20",  # ISO date shape
                "ts": "2026-07-20T12:00:00",  # ISO datetime shape
                "version_like": "2026-07-2",  # 9 chars, not ISO
                "hyphenated": "hi-jk-lm-no",  # 10 chars with '-', not ISO
                "date_with_space": "2026-07-20 12:00",  # legacy datetime shape
            }
        ]
        backend.save_dataset("date_like_strings", data)
        loaded = backend.load_dataset("date_like_strings")

        for key, value in data[0].items():
            loaded_value = loaded[0][key]
            assert isinstance(loaded_value, str), (
                f"{key!r} was coerced to {type(loaded_value).__name__}"
            )
            assert loaded_value == value

    def test_date_round_trip(self, backend):
        """A saved date object must come back as a date."""
        data = [{"d": date(2026, 7, 20)}]
        backend.save_dataset("date_round_trip", data)
        loaded = backend.load_dataset("date_round_trip")
        assert loaded[0]["d"] == date(2026, 7, 20)
        assert type(loaded[0]["d"]) is date

    def test_datetime_round_trip(self, backend):
        """A saved datetime object must come back as a datetime."""
        data = [
            {
                "naive": datetime(2026, 7, 20, 12, 0, 0),
                "with_micros": datetime(2026, 7, 20, 12, 0, 0, 123456),
                "aware": datetime.fromisoformat("2026-07-20T12:00:00+05:30"),
            }
        ]
        backend.save_dataset("datetime_round_trip", data)
        loaded = backend.load_dataset("datetime_round_trip")

        for key, value in data[0].items():
            assert loaded[0][key] == value, key
            assert isinstance(loaded[0][key], datetime), key

    def test_nested_date_like_strings(self, backend):
        """Coercion must not happen inside nested dicts or lists either."""
        data = [
            {
                "outer": {"created": "2026-07-20", "id": "hi-jk-lm-no"},
                "history": [{"ts": "2026-07-20T12:00:00"}],
                "dates": ["2026-07-20", "2026-07-21"],
            }
        ]
        backend.save_dataset("nested_date_like", data)
        loaded = backend.load_dataset("nested_date_like")

        assert loaded[0]["outer"]["created"] == "2026-07-20"
        assert loaded[0]["outer"]["id"] == "hi-jk-lm-no"
        assert loaded[0]["history"][0]["ts"] == "2026-07-20T12:00:00"
        assert loaded[0]["dates"] == ["2026-07-20", "2026-07-21"]

    def test_nested_typed_dates_round_trip(self, backend):
        """date / datetime objects nested in dicts and lists round trip."""
        data = [
            {
                "outer": {"created": date(2026, 7, 20)},
                "history": [{"ts": datetime(2026, 7, 20, 12, 0)}],
            }
        ]
        backend.save_dataset("nested_typed_dates", data)
        loaded = backend.load_dataset("nested_typed_dates")

        assert loaded[0]["outer"]["created"] == date(2026, 7, 20)
        assert loaded[0]["history"][0]["ts"] == datetime(2026, 7, 20, 12, 0)

    def test_legacy_file_exact_iso_shapes_still_coerce(self, backend, temp_dir):
        """Files written before tagging coerce only anchored ISO 8601 shapes."""
        datasets_dir = Path(temp_dir) / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        with open(datasets_dir / "legacy.jsonl", "w", encoding="utf-8") as f:
            f.write('{"d": "2026-07-20", "ts": "2026-07-20T12:00:00"}\n')
            f.write('{"not_iso": "hi-jk-lm-no", "partial": "2026-07-2"}\n')
            f.write('{"tz": "2026-07-20T12:00:00+05:30"}\n')

        loaded = backend.load_dataset("legacy")
        assert loaded[0]["d"] == date(2026, 7, 20)
        assert loaded[0]["ts"] == datetime(2026, 7, 20, 12, 0, 0)
        assert loaded[1]["not_iso"] == "hi-jk-lm-no"
        assert loaded[1]["partial"] == "2026-07-2"
        assert loaded[2]["tz"] == datetime.fromisoformat("2026-07-20T12:00:00+05:30")

    def test_dict_with_discriminator_key_round_trips(self, backend):
        """A user dict that already contains __ragas_type__ must round trip verbatim."""
        user_dict = {"__ragas_type__": "date", "value": "not-a-real-date"}
        data = [{"payload": user_dict, "name": "x"}]
        backend.save_dataset("tag_collision", data)
        loaded = backend.load_dataset("tag_collision")

        assert loaded[0]["payload"] == user_dict
        assert isinstance(loaded[0]["payload"], dict)


if __name__ == "__main__":
    # Run some quick tests to see JSONL capabilities
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            backend: LocalJSONLBackend = LocalJSONLBackend(tmp_dir)

            # Test nested data
            test_nested_data: list[dict[str, t.Any]] = [
                {"id": 1, "metadata": {"score": 0.85, "tags": ["test"]}}
            ]
            backend.save_dataset("debug_nested", test_nested_data)
            loaded = backend.load_dataset("debug_nested")

            print("=== Nested Data Test ===")
            print(f"Original: {test_nested_data[0]['metadata']}")
            print(f"Loaded: {loaded[0]['metadata']}")
            print(
                f"Types: {type(test_nested_data[0]['metadata'])} -> {type(loaded[0]['metadata'])}"
            )

            print_jsonl_content(backend, "datasets", "debug_nested")

        except ImportError as e:
            print(f"Expected ImportError: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
