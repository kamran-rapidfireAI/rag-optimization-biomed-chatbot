"""Embedding cache for BioRAG Bench."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

from biorag.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """Disk-based embedding cache keyed by (model, text_hash)."""

    def __init__(
        self,
        cache_dir: Path | str,
        model_name: str,
    ) -> None:
        """
        Initialize embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings
            model_name: Model name for cache key prefix
        """
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self._model_hash = self._hash_string(model_name)[:8]

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Stats
        self.hits = 0
        self.misses = 0

    def _hash_string(self, text: str) -> str:
        """Create a hash of a text string."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for a text."""
        text_hash = self._hash_string(text)
        return f"{self._model_hash}_{text_hash}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for a cache key."""
        # Use subdirectories to avoid too many files in one directory
        subdir = cache_key[:2]
        return self.cache_dir / subdir / f"{cache_key}.pkl"

    def get(self, text: str) -> list[float] | None:
        """
        Get cached embedding for text.

        Args:
            text: Text to look up

        Returns:
            Cached embedding or None if not found
        """
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    embedding = pickle.load(f)
                self.hits += 1
                return embedding
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        self.misses += 1
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        """
        Store embedding in cache.

        Args:
            text: Text that was embedded
            embedding: Embedding vector to cache
        """
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def get_batch(self, texts: list[str]) -> tuple[list[list[float] | None], list[int]]:
        """
        Get cached embeddings for multiple texts.

        Args:
            texts: List of texts to look up

        Returns:
            Tuple of (embeddings with None for misses, indices of misses)
        """
        embeddings: list[list[float] | None] = []
        miss_indices: list[int] = []

        for i, text in enumerate(texts):
            cached = self.get(text)
            embeddings.append(cached)
            if cached is None:
                miss_indices.append(i)

        return embeddings, miss_indices

    def put_batch(self, texts: list[str], embeddings: list[list[float]]) -> None:
        """
        Store multiple embeddings in cache.

        Args:
            texts: List of texts that were embedded
            embeddings: List of embedding vectors
        """
        for text, embedding in zip(texts, embeddings):
            self.put(text, embedding)

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def clear(self) -> None:
        """Clear all cached embeddings."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.hits = 0
        self.misses = 0
        logger.info(f"Cleared embedding cache: {self.cache_dir}")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "model": self.model_name,
            "cache_dir": str(self.cache_dir),
        }

