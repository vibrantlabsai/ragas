"""Unit tests for embeddings caching functionality."""

from unittest.mock import MagicMock

import pytest

from ragas.cache import DiskCacheBackend
from ragas.embeddings import embedding_factory


def test_embeddings_cache_hit(tmp_path):
    """Test that embeddings caching works."""
    cache = DiskCacheBackend(cache_dir=str(tmp_path / "cache"))

    # Mock client
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
    )

    embedder = embedding_factory("openai", client=mock_client, cache=cache)

    # First call - should call API
    emb1 = embedder.embed_text("test text")
    assert mock_client.embeddings.create.call_count == 1

    # Second call - should hit cache
    emb2 = embedder.embed_text("test text")
    assert mock_client.embeddings.create.call_count == 1  # Still 1!
    assert emb1 == emb2


def test_embeddings_cache_miss_different_text(tmp_path):
    """Test that different texts don't hit cache."""
    cache = DiskCacheBackend(cache_dir=str(tmp_path / "cache"))

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
    )

    embedder = embedding_factory("openai", client=mock_client, cache=cache)

    # Two different texts
    embedder.embed_text("text 1")
    embedder.embed_text("text 2")

    # Should call API twice
    assert mock_client.embeddings.create.call_count == 2


def test_embeddings_cache_batch_benefits(tmp_path):
    """Test that batch embeddings benefit from single-text cache."""
    cache = DiskCacheBackend(cache_dir=str(tmp_path / "cache"))

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
    )

    embedder = embedding_factory("openai", client=mock_client, cache=cache)

    # Embed single text first
    embedder.embed_text("text 1")
    assert mock_client.embeddings.create.call_count == 1

    # Embed batch with same text - should hit cache for the one we've seen
    embedder.embed_texts(["text 1", "text 2"])

    # Should only call once more for "text 2" (text 1 was cached)
    assert mock_client.embeddings.create.call_count == 2


@pytest.mark.asyncio
async def test_embeddings_cache_async(tmp_path):
    """Test that async embeddings caching works."""
    cache = DiskCacheBackend(cache_dir=str(tmp_path / "cache"))

    mock_client = MagicMock()

    # Mock async method
    async def mock_create(*args, **kwargs):
        return MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])])

    mock_client.embeddings.create = mock_create

    embedder = embedding_factory("openai", client=mock_client, cache=cache)

    # First call
    emb1 = await embedder.aembed_text("async text")

    # Second call - should hit cache
    emb2 = await embedder.aembed_text("async text")

    assert emb1 == emb2


def test_embeddings_no_cache_parameter(tmp_path):
    """Test that embeddings work without cache parameter (backward compatibility)."""
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
    )

    # Should work without cache
    embedder = embedding_factory("openai", client=mock_client)
    result = embedder.embed_text("test")

    assert result == [0.1, 0.2, 0.3]


def test_cache_persistence_across_sessions(tmp_path):
    """Test that cache persists across different Python sessions (instances)."""
    cache_dir = str(tmp_path / "cache")

    # Session 1: Create embedder, make call, cache it
    cache1 = DiskCacheBackend(cache_dir=cache_dir)
    mock_client1 = MagicMock()
    mock_client1.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
    )

    embedder1 = embedding_factory("openai", client=mock_client1, cache=cache1)
    embedder1.embed_text("persistent text")
    assert mock_client1.embeddings.create.call_count == 1

    # Session 2: New cache instance, same directory
    cache2 = DiskCacheBackend(cache_dir=cache_dir)
    mock_client2 = MagicMock()
    mock_client2.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.9, 0.9, 0.9])]
    )

    embedder2 = embedding_factory("openai", client=mock_client2, cache=cache2)
    result2 = embedder2.embed_text("persistent text")

    # Should hit cache from session 1, not call API
    assert mock_client2.embeddings.create.call_count == 0
    assert result2 == [0.1, 0.2, 0.3]  # From cache, not the new mock value
