"""Tests for the experiment module."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from ragas.backends.inmemory import InMemoryBackend
from ragas.dataset import Dataset
from ragas.experiment import Experiment, experiment, version_experiment
from ragas.utils import find_git_root, memorable_names


# Test data models
class SampleDataRow(BaseModel):
    question: str
    answer: str
    score: float


class ExperimentResultRow(BaseModel):
    question: str
    processed_answer: str
    sentiment: str
    processing_time: float


# Test fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_git_repo(temp_dir):
    """Create a mock git repository."""
    git_dir = temp_dir / ".git"
    git_dir.mkdir()

    # Mock git.Repo
    mock_repo = MagicMock()
    mock_repo.is_dirty.return_value = False
    mock_repo.head.commit.hexsha = "abc123def456"
    mock_repo.git.add = MagicMock()
    mock_repo.index.commit = MagicMock()
    mock_repo.create_head = MagicMock()

    with patch("git.Repo", return_value=mock_repo):
        yield mock_repo, temp_dir


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    backend = InMemoryBackend()
    dataset = Dataset(
        name="test_dataset",
        data_model=SampleDataRow,
        backend=backend,
        data=[
            SampleDataRow(
                question="What is Python?", answer="A programming language", score=0.9
            ),
            SampleDataRow(
                question="What is AI?", answer="Artificial Intelligence", score=0.8
            ),
            SampleDataRow(
                question="What is ML?", answer="Machine Learning", score=0.85
            ),
        ],
    )
    return dataset


@pytest.fixture
def experiment_backend():
    """Create a backend for experiments."""
    return InMemoryBackend()


# Test classes
class TestExperiment:
    """Test the Experiment class."""

    def test_experiment_inheritance(self):
        """Test that Experiment properly inherits from DataTable."""
        assert hasattr(Experiment, "DATATABLE_TYPE")
        assert Experiment.DATATABLE_TYPE == "Experiment"

    def test_experiment_creation(self, experiment_backend):
        """Test creating an Experiment instance."""
        experiment = Experiment(
            name="test_experiment",
            data_model=ExperimentResultRow,
            backend=experiment_backend,
        )

        assert experiment.name == "test_experiment"
        assert experiment.backend == experiment_backend
        assert len(experiment) == 0


class TestVersionExperiment:
    """Test the version_experiment function."""

    def test_version_experiment_no_changes(self, mock_git_repo):
        """Test version_experiment when there are no changes."""
        mock_repo, temp_dir = mock_git_repo

        # Mock that repo is clean
        mock_repo.is_dirty.return_value = False

        with patch("ragas.utils.find_git_root", return_value=temp_dir):
            commit_hash = version_experiment("test_experiment")

        assert commit_hash == "abc123def456"
        mock_repo.is_dirty.assert_called()
        mock_repo.create_head.assert_called_with(
            "ragas/test_experiment", "abc123def456"
        )

    def test_version_experiment_with_changes(self, mock_git_repo):
        """Test version_experiment when there are changes to commit."""
        mock_repo, temp_dir = mock_git_repo

        # Mock that repo is dirty
        mock_repo.is_dirty.return_value = True

        # Mock commit object
        mock_commit = MagicMock()
        mock_commit.hexsha = "new123commit456"
        mock_repo.index.commit.return_value = mock_commit

        with patch("ragas.utils.find_git_root", return_value=temp_dir):
            commit_hash = version_experiment("test_experiment")

        assert commit_hash == "new123commit456"
        mock_repo.git.add.assert_called_with("-u")
        mock_repo.index.commit.assert_called_once()

    def test_version_experiment_with_custom_message(self, mock_git_repo):
        """Test version_experiment with custom commit message."""
        mock_repo, temp_dir = mock_git_repo
        mock_repo.is_dirty.return_value = True

        mock_commit = MagicMock()
        mock_commit.hexsha = "custom123commit456"
        mock_repo.index.commit.return_value = mock_commit

        with patch("ragas.utils.find_git_root", return_value=temp_dir):
            version_experiment(
                "test_experiment", commit_message="Custom experiment message"
            )

        mock_repo.index.commit.assert_called_with("Custom experiment message")

    def test_version_experiment_stage_all(self, mock_git_repo):
        """Test version_experiment with stage_all=True."""
        mock_repo, temp_dir = mock_git_repo
        mock_repo.is_dirty.return_value = True

        mock_commit = MagicMock()
        mock_commit.hexsha = "staged123commit456"
        mock_repo.index.commit.return_value = mock_commit

        with patch("ragas.utils.find_git_root", return_value=temp_dir):
            version_experiment("test_experiment", stage_all=True)

        mock_repo.git.add.assert_called_with(".")

    def test_version_experiment_no_branch_creation(self, mock_git_repo):
        """Test version_experiment with create_branch=False."""
        mock_repo, temp_dir = mock_git_repo

        with patch("ragas.utils.find_git_root", return_value=temp_dir):
            version_experiment("test_experiment", create_branch=False)

        mock_repo.create_head.assert_not_called()

    def test_find_git_root_error_handling(self, temp_dir):
        """Test that find_git_root raises ValueError when no git repo found."""
        with pytest.raises(ValueError, match="No git repository found"):
            find_git_root(temp_dir)

    def test_version_experiment_missing_gitpython(self, temp_dir):
        """Test that version_experiment provides helpful error when GitPython is not installed."""
        with patch("ragas.utils.find_git_root", return_value=temp_dir):
            with patch.dict("sys.modules", {"git": None}):
                with pytest.raises(ImportError, match="uv pip install ragas\\[git\\]"):
                    version_experiment("test_experiment")


class TestExperimentDecorator:
    """Test the experiment decorator."""

    @pytest.mark.asyncio
    async def test_simple_async_experiment(self, sample_dataset, experiment_backend):
        """Test a simple async experiment function."""

        @experiment(experiment_model=ExperimentResultRow, backend=experiment_backend)
        async def simple_experiment(row: SampleDataRow) -> ExperimentResultRow:
            return ExperimentResultRow(
                question=row.question,
                processed_answer=row.answer.upper(),
                sentiment="positive",
                processing_time=0.1,
            )

        # Test that decorator creates proper wrapper
        assert hasattr(simple_experiment, "arun")
        assert hasattr(simple_experiment, "__call__")

        # Test calling the wrapped function directly
        test_row = SampleDataRow(question="Test?", answer="test answer", score=0.5)
        result = await simple_experiment(test_row)

        assert isinstance(result, ExperimentResultRow)
        assert result.processed_answer == "TEST ANSWER"
        assert result.sentiment == "positive"

    @pytest.mark.asyncio
    async def test_experiment_arun(self, sample_dataset, experiment_backend):
        """Test running experiment against a dataset."""

        @experiment(experiment_model=ExperimentResultRow, backend=experiment_backend)
        async def test_experiment(row: SampleDataRow) -> ExperimentResultRow:
            return ExperimentResultRow(
                question=row.question,
                processed_answer=row.answer.lower(),
                sentiment="neutral",
                processing_time=0.05,
            )

        # Mock memorable_names to return predictable name
        with patch(
            "ragas.utils.memorable_names.generate_unique_name",
            return_value="test_experiment_name",
        ):
            experiment_result = await test_experiment.arun(sample_dataset)

        assert isinstance(experiment_result, Experiment)
        assert experiment_result.name == "test_experiment_name"
        assert len(experiment_result) == 3  # Should have processed all 3 items

    @pytest.mark.asyncio
    async def test_experiment_with_name_prefix(
        self, sample_dataset, experiment_backend
    ):
        """Test experiment decorator with name prefix."""

        @experiment(
            experiment_model=ExperimentResultRow,
            backend=experiment_backend,
            name_prefix="prefix",
        )
        async def prefixed_experiment(row: SampleDataRow) -> ExperimentResultRow:
            return ExperimentResultRow(
                question=row.question,
                processed_answer=row.answer,
                sentiment="neutral",
                processing_time=0.01,
            )

        with patch(
            "ragas.utils.memorable_names.generate_unique_name",
            return_value="random_name",
        ):
            experiment_result = await prefixed_experiment.arun(sample_dataset)

        assert experiment_result.name == "prefix-random_name"

    @pytest.mark.asyncio
    async def test_experiment_with_custom_name(
        self, sample_dataset, experiment_backend
    ):
        """Test experiment with custom name."""

        @experiment(experiment_model=ExperimentResultRow, backend=experiment_backend)
        async def custom_named_experiment(row: SampleDataRow) -> ExperimentResultRow:
            return ExperimentResultRow(
                question=row.question,
                processed_answer=row.answer,
                sentiment="positive",
                processing_time=0.02,
            )

        experiment_result = await custom_named_experiment.arun(
            sample_dataset, name="my_custom_experiment"
        )

        assert experiment_result.name == "my_custom_experiment"

    def test_sync_experiment_function(self, experiment_backend):
        """Test that sync functions work with the experiment decorator."""

        @experiment(experiment_model=ExperimentResultRow, backend=experiment_backend)
        def sync_experiment(row: SampleDataRow) -> ExperimentResultRow:
            return ExperimentResultRow(
                question=row.question,
                processed_answer=row.answer.upper(),
                sentiment="positive",
                processing_time=0.0,
            )

        # Test that we can call it synchronously within async context
        test_row = SampleDataRow(question="Sync test?", answer="sync answer", score=0.7)

        async def test_sync_call():
            result = await sync_experiment(test_row)
            return result

        result = asyncio.run(test_sync_call())
        assert isinstance(result, ExperimentResultRow)
        assert result.processed_answer == "SYNC ANSWER"

    @pytest.mark.asyncio
    async def test_experiment_error_handling(self, sample_dataset, experiment_backend):
        """Test that experiment handles individual task failures gracefully."""

        @experiment(experiment_model=ExperimentResultRow, backend=experiment_backend)
        async def failing_experiment(row: SampleDataRow) -> ExperimentResultRow:
            if "AI" in row.question:  # Fail on the AI question
                raise ValueError("Test error")
            return ExperimentResultRow(
                question=row.question,
                processed_answer=row.answer,
                sentiment="neutral",
                processing_time=0.01,
            )

        # Should continue processing other items even if some fail
        with patch(
            "ragas.utils.memorable_names.generate_unique_name",
            return_value="error_test",
        ):
            experiment_result = await failing_experiment.arun(sample_dataset)

        # Should have 2 successful results (3 items - 1 failure)
        assert len(experiment_result) == 2

    @pytest.mark.asyncio
    async def test_experiment_with_no_model(self, sample_dataset, experiment_backend):
        """Test experiment without specifying a model."""

        @experiment(backend=experiment_backend)
        async def untyped_experiment(row: SampleDataRow) -> dict:
            return {"question": row.question, "answer": row.answer, "processed": True}

        with patch(
            "ragas.utils.memorable_names.generate_unique_name",
            return_value="untyped_test",
        ):
            experiment_result = await untyped_experiment.arun(sample_dataset)

        assert isinstance(experiment_result, Experiment)
        assert len(experiment_result) == 3


class TestMemorableNames:
    """Test the memorable names functionality."""

    def test_memorable_names_generation(self):
        """Test that memorable names are generated correctly."""
        name = memorable_names.generate_name()
        assert "_" in name
        parts = name.split("_", 1)  # Split on first underscore only
        assert len(parts) == 2
        assert parts[0] in memorable_names.adjectives
        assert parts[1] in memorable_names.scientists

    def test_unique_name_generation(self):
        """Test that unique names are generated."""
        # Create a fresh instance to avoid state from other tests
        from ragas.utils import MemorableNames

        generator = MemorableNames()

        names = [generator.generate_unique_name() for _ in range(10)]
        assert len(set(names)) == 10  # All names should be unique

    def test_unique_names_batch_generation(self):
        """Test batch generation of unique names."""
        from ragas.utils import MemorableNames

        generator = MemorableNames()

        names = generator.generate_unique_names(5)
        assert len(names) == 5
        assert len(set(names)) == 5  # All should be unique


class TestUtilityFunctions:
    """Test utility functions added to ragas.utils."""

    def test_find_git_root_with_git_repo(self, temp_dir):
        """Test find_git_root finds git repository correctly."""
        # Create a nested directory structure with .git at the top
        git_dir = temp_dir / ".git"
        git_dir.mkdir()

        nested_dir = temp_dir / "nested" / "deeply" / "nested"
        nested_dir.mkdir(parents=True)

        # Should find git root from nested directory
        found_root = find_git_root(nested_dir)
        # Use resolve() to handle symlinks and get canonical path
        assert found_root.resolve() == temp_dir.resolve()

    def test_find_git_root_current_dir(self):
        """Test find_git_root uses current directory when no path provided."""
        # This should find the actual git root of the ragas project
        try:
            root = find_git_root()
            assert isinstance(root, Path)
            assert (root / ".git").exists()
        except ValueError:
            # If we're not in a git repo, that's expected
            pass

    def test_find_git_root_no_repo_error(self, temp_dir):
        """Test find_git_root raises error when no git repo found."""
        with pytest.raises(ValueError, match="No git repository found"):
            find_git_root(temp_dir)
