"""Base embedder interface for BioRAG Bench."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, model: str, dimension: int | None = None) -> None:
        """
        Initialize embedder.

        Args:
            model: Model name/identifier
            dimension: Embedding dimension (if known)
        """
        self.model = model
        self.dimension = dimension

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        pass

    def embed_batch(
        self, texts: list[str], batch_size: int = 100
    ) -> list[list[float]]:
        """
        Embed texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.embed_documents(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension."""
        if self.dimension is None:
            # Compute dimension by embedding a sample text
            sample = self.embed_query("test")
            self.dimension = len(sample)
        return self.dimension

