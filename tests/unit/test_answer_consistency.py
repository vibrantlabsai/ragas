from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._answer_consistency import AnswerConsistency
from ragas.metrics.base import MetricType


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    embeddings = MagicMock()
    # Mock the aembed_text method to return normalized embeddings
    embeddings.aembed_text = AsyncMock()
    return embeddings


def test_answer_consistency_init():
    """Test AnswerConsistency initialization."""
    metric = AnswerConsistency()
    assert metric.name == "answer_consistency"
    assert metric._required_columns == {
        MetricType.SINGLE_TURN: {"multi_responses"}
    }
    assert metric.min_responses == 2


def test_answer_consistency_init_invalid_min_responses():
    """Test AnswerConsistency initialization with invalid min_responses."""
    with pytest.raises(ValueError, match="min_responses must be at least 2"):
        AnswerConsistency(min_responses=1)


@pytest.mark.asyncio
async def test_answer_consistency_perfect_similarity(mock_embeddings):
    """Test with identical responses (perfect consistency)."""
    metric = AnswerConsistency(embeddings=mock_embeddings)
    
    # Mock identical embeddings for identical responses
    identical_embedding = np.array([1.0, 0.0, 0.0])
    mock_embeddings.aembed_text.return_value = identical_embedding
    
    sample = SingleTurnSample(
        multi_responses=[
            "Paris is the capital of France.",
            "Paris is the capital of France.",
            "Paris is the capital of France.",
        ]
    )
    
    score = await metric._single_turn_ascore(sample, None)
    
    assert isinstance(score, float)
    assert score == pytest.approx(1.0, abs=0.01)
    assert mock_embeddings.aembed_text.call_count == 3


@pytest.mark.asyncio
async def test_answer_consistency_varied_similarity(mock_embeddings):
    """Test with semantically similar but not identical responses."""
    metric = AnswerConsistency(embeddings=mock_embeddings)
    
    # Mock embeddings with high but not perfect similarity
    embeddings_list = [
        np.array([1.0, 0.0, 0.0]),  # Response 1
        np.array([0.95, 0.1, 0.0]),  # Response 2 (similar)
        np.array([0.9, 0.2, 0.0]),   # Response 3 (similar)
    ]
    
    mock_embeddings.aembed_text.side_effect = embeddings_list
    
    sample = SingleTurnSample(
        multi_responses=[
            "Paris is the capital of France.",
            "The capital city of France is Paris.",
            "France's capital is Paris.",
        ]
    )
    
    score = await metric._single_turn_ascore(sample, None)
    
    assert isinstance(score, float)
    assert 0.8 < score < 1.0  # High similarity but not perfect
    assert mock_embeddings.aembed_text.call_count == 3


@pytest.mark.asyncio
async def test_answer_consistency_low_similarity(mock_embeddings):
    """Test with very different responses (low consistency)."""
    metric = AnswerConsistency(embeddings=mock_embeddings)
    
    # Mock embeddings with low similarity (orthogonal vectors)
    embeddings_list = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    
    mock_embeddings.aembed_text.side_effect = embeddings_list
    
    sample = SingleTurnSample(
        multi_responses=[
            "Paris is the capital of France.",
            "The sky is blue.",
            "Elephants are large animals.",
        ]
    )
    
    score = await metric._single_turn_ascore(sample, None)
    
    assert isinstance(score, float)
    assert score == pytest.approx(0.0, abs=0.01)  # Orthogonal = 0 similarity
    assert mock_embeddings.aembed_text.call_count == 3


@pytest.mark.asyncio
async def test_answer_consistency_minimum_responses(mock_embeddings):
    """Test with exactly minimum required responses."""
    metric = AnswerConsistency(embeddings=mock_embeddings)
    
    # Mock embeddings
    embeddings_list = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.9, 0.1, 0.0]),
    ]
    mock_embeddings.aembed_text.side_effect = embeddings_list
    
    sample = SingleTurnSample(
        multi_responses=[
            "Paris is the capital of France.",
            "The capital of France is Paris.",
        ]
    )
    
    score = await metric._single_turn_ascore(sample, None)
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert mock_embeddings.aembed_text.call_count == 2


@pytest.mark.asyncio
async def test_answer_consistency_missing_multi_responses(mock_embeddings):
    """Test error when multi_responses is missing."""
    metric = AnswerConsistency(embeddings=mock_embeddings)
    
    sample = SingleTurnSample(response="Single response")
    
    with pytest.raises(ValueError, match="multi_responses is missing or empty"):
        await metric._single_turn_ascore(sample, None)


@pytest.mark.asyncio
async def test_answer_consistency_empty_multi_responses(mock_embeddings):
    """Test error when multi_responses is empty."""
    metric = AnswerConsistency(embeddings=mock_embeddings)
    
    sample = SingleTurnSample(multi_responses=[])
    
    with pytest.raises(ValueError, match="multi_responses is missing or empty"):
        await metric._single_turn_ascore(sample, None)


@pytest.mark.asyncio
async def test_answer_consistency_insufficient_responses(mock_embeddings):
    """Test error when multi_responses has fewer than min_responses."""
    metric = AnswerConsistency(embeddings=mock_embeddings)
    
    sample = SingleTurnSample(multi_responses=["Only one response"])
    
    with pytest.raises(ValueError, match="must contain at least 2 responses"):
        await metric._single_turn_ascore(sample, None)


@pytest.mark.asyncio
async def test_answer_consistency_empty_strings(mock_embeddings):
    """Test handling of empty strings in responses."""
    metric = AnswerConsistency(embeddings=mock_embeddings)
    
    # Mock embeddings for empty strings (replaced with space)
    embeddings_list = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.95, 0.1, 0.0]),
    ]
    mock_embeddings.aembed_text.side_effect = embeddings_list
    
    sample = SingleTurnSample(multi_responses=["", "Some response"])
    
    score = await metric._single_turn_ascore(sample, None)
    
    assert isinstance(score, float)
    # Verify empty string was replaced with space
    assert mock_embeddings.aembed_text.call_args_list[0][0][0] == " "


@pytest.mark.asyncio
async def test_answer_consistency_legacy_embeddings_interface(mock_embeddings):
    """Test with legacy embeddings interface (embed_text instead of aembed_text)."""
    # Remove aembed_text to simulate legacy interface
    delattr(mock_embeddings, "aembed_text")
    mock_embeddings.embed_text = AsyncMock()
    
    embeddings_list = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.95, 0.1, 0.0]),
    ]
    mock_embeddings.embed_text.side_effect = embeddings_list
    
    metric = AnswerConsistency(embeddings=mock_embeddings)
    
    sample = SingleTurnSample(
        multi_responses=[
            "Paris is the capital of France.",
            "The capital of France is Paris.",
        ]
    )
    
    score = await metric._single_turn_ascore(sample, None)
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert mock_embeddings.embed_text.call_count == 2


@pytest.mark.asyncio
async def test_answer_consistency_ascore_from_dict(mock_embeddings):
    """Test async score calculation from dictionary row."""
    metric = AnswerConsistency(embeddings=mock_embeddings)
    
    embeddings_list = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.9, 0.1, 0.0]),
    ]
    mock_embeddings.aembed_text.side_effect = embeddings_list
    
    row = {
        "multi_responses": [
            "Paris is the capital of France.",
            "The capital of France is Paris.",
        ]
    }
    
    score = await metric._ascore(row, None)
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert mock_embeddings.aembed_text.call_count == 2


@pytest.mark.asyncio
async def test_answer_consistency_no_embeddings():
    """Test error when embeddings are not set."""
    metric = AnswerConsistency()  # No embeddings provided
    
    sample = SingleTurnSample(
        multi_responses=["Response 1", "Response 2"]
    )
    
    with pytest.raises(AssertionError, match="requires embeddings to be set"):
        await metric._single_turn_ascore(sample, None)


@pytest.mark.asyncio
async def test_answer_consistency_many_responses(mock_embeddings):
    """Test with many responses (tests pairwise combinations)."""
    metric = AnswerConsistency(embeddings=mock_embeddings)
    
    # Create 5 similar embeddings
    embeddings_list = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.95, 0.05, 0.0]),
        np.array([0.9, 0.1, 0.0]),
        np.array([0.92, 0.08, 0.0]),
        np.array([0.88, 0.12, 0.0]),
    ]
    mock_embeddings.aembed_text.side_effect = embeddings_list
    
    sample = SingleTurnSample(
        multi_responses=[
            "Response 1",
            "Response 2",
            "Response 3",
            "Response 4",
            "Response 5",
        ]
    )
    
    score = await metric._single_turn_ascore(sample, None)
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # With 5 responses, we have C(5,2) = 10 pairwise comparisons
    assert mock_embeddings.aembed_text.call_count == 5