"""Regression tests for #2814: local backends must not let a dataset/experiment
name escape the configured storage root via path components."""

import tempfile
from pathlib import Path

import pytest

from ragas.backends.local_csv import LocalCSVBackend
from ragas.backends.local_jsonl import LocalJSONLBackend
from ragas.backends.utils import validate_local_name

ESCAPING_NAMES = [
    "../escape",
    "../../escape",
    "..\\escape",
    "sub/child",
    "/abs",
    ".",
    "..",
    "",
    "   ",
]

VALID_NAMES = ["dataset-1_v1.0", "experiment", "my_data", "run.2024-01-01"]


@pytest.fixture
def root():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.mark.parametrize("name", ESCAPING_NAMES)
def test_validate_local_name_rejects_path_escaping(name):
    with pytest.raises(ValueError):
        validate_local_name(name)


@pytest.mark.parametrize("name", VALID_NAMES)
def test_validate_local_name_accepts_plain_identifiers(name):
    assert validate_local_name(name) == name


@pytest.mark.parametrize("backend_cls", [LocalCSVBackend, LocalJSONLBackend])
@pytest.mark.parametrize("name", ["../../escape", "..\\escape", "sub/child"])
def test_save_dataset_rejects_traversal(backend_cls, name, root):
    backend = backend_cls(root)
    with pytest.raises(ValueError):
        backend.save_dataset(name, [{"x": 1}])

    # Nothing was written outside the storage root.
    root_path = Path(root).resolve()
    for created in Path(root).parent.rglob("escape*"):
        assert root_path in created.resolve().parents, (
            f"file {created} was written outside the storage root"
        )


@pytest.mark.parametrize("backend_cls", [LocalCSVBackend, LocalJSONLBackend])
@pytest.mark.parametrize("name", ["../../escape", "sub/child"])
def test_load_dataset_rejects_traversal(backend_cls, name, root):
    backend = backend_cls(root)
    with pytest.raises(ValueError):
        backend.load_dataset(name)


@pytest.mark.parametrize(
    "backend_cls,ext",
    [(LocalCSVBackend, "csv"), (LocalJSONLBackend, "jsonl")],
)
def test_valid_name_still_roundtrips(backend_cls, ext, root):
    backend = backend_cls(root)
    backend.save_dataset("dataset-1_v1.0", [{"x": 1}])
    written = Path(root) / "datasets" / f"dataset-1_v1.0.{ext}"
    assert written.exists()
    loaded = backend.load_dataset("dataset-1_v1.0")
    assert len(loaded) == 1 and "x" in loaded[0]
