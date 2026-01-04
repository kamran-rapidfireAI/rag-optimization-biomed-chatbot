"""Unit tests for embedding modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from biorag.embeddings.cache import EmbeddingCache


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    @pytest.fixture
    def cache(self, temp_dir: Path) -> EmbeddingCache:
        """Create a cache instance."""
        return EmbeddingCache(
            cache_dir=temp_dir / "embeddings",
            model_name="test-model",
        )

    def test_cache_miss(self, cache: EmbeddingCache) -> None:
        """Test cache miss returns None."""
        result = cache.get("some text")
        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_cache_put_get(self, cache: EmbeddingCache) -> None:
        """Test storing and retrieving embeddings."""
        text = "This is test text"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        cache.put(text, embedding)
        result = cache.get(text)

        assert result == embedding
        assert cache.hits == 1

    def test_cache_batch(self, cache: EmbeddingCache) -> None:
        """Test batch operations."""
        texts = ["text1", "text2", "text3"]
        embeddings = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]

        # Put batch
        cache.put_batch(texts, embeddings)

        # Get batch
        results, misses = cache.get_batch(texts + ["text4"])

        assert len(results) == 4
        assert results[0] == embeddings[0]
        assert results[3] is None
        assert misses == [3]

    def test_cache_hit_rate(self, cache: EmbeddingCache) -> None:
        """Test hit rate calculation."""
        cache.put("text1", [0.1])

        cache.get("text1")  # Hit
        cache.get("text2")  # Miss
        cache.get("text1")  # Hit

        assert cache.hits == 2
        assert cache.misses == 1
        assert cache.hit_rate == pytest.approx(2 / 3)

    def test_cache_clear(self, cache: EmbeddingCache) -> None:
        """Test clearing cache."""
        cache.put("text1", [0.1])
        assert cache.get("text1") is not None

        cache.clear()
        assert cache.get("text1") is None
        assert cache.hits == 0
        assert cache.misses == 1  # From the get after clear

    def test_cache_stats(self, cache: EmbeddingCache) -> None:
        """Test stats retrieval."""
        cache.put("text1", [0.1])
        cache.get("text1")
        cache.get("text2")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["model"] == "test-model"


class TestEmbedderInterface:
    """Tests for Embedder interface."""

    def test_embedder_requires_implementation(self) -> None:
        """Test that abstract methods must be implemented."""
        from biorag.embeddings.base import Embedder

        # Can't instantiate abstract class
        with pytest.raises(TypeError):
            Embedder(model="test")  # type: ignore

    def test_embed_batch_uses_embed_documents(self) -> None:
        """Test that embed_batch correctly batches calls."""
        from biorag.embeddings.base import Embedder

        class TestEmbedder(Embedder):
            def __init__(self) -> None:
                super().__init__(model="test", dimension=3)
                self.call_count = 0

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                self.call_count += 1
                return [[0.1, 0.2, 0.3] for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

        embedder = TestEmbedder()
        texts = [f"text{i}" for i in range(25)]

        result = embedder.embed_batch(texts, batch_size=10)

        assert len(result) == 25
        assert embedder.call_count == 3  # 10 + 10 + 5

