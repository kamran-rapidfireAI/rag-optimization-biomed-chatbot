"""OpenAI embeddings for BioRAG Bench."""

from __future__ import annotations

import os

from openai import OpenAI

from biorag.embeddings.base import Embedder
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings provider."""

    # Known embedding dimensions
    DIMENSIONS = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: str | None = None,
        batch_size: int = 100,
    ) -> None:
        """
        Initialize OpenAI embedder.

        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            batch_size: Batch size for embedding requests
        """
        dimension = self.DIMENSIONS.get(model)
        super().__init__(model=model, dimension=dimension)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")

        self.batch_size = batch_size
        self._client = OpenAI(api_key=self.api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents using OpenAI API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Clean texts (OpenAI doesn't like empty strings)
        cleaned_texts = [t.strip() if t else " " for t in texts]

        try:
            response = self._client.embeddings.create(
                input=cleaned_texts,
                model=self.model,
            )

            # Sort by index to ensure correct ordering
            embeddings = sorted(response.data, key=lambda x: x.index)
            return [e.embedding for e in embeddings]

        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query using OpenAI API.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []

    def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """
        Embed texts in batches to handle rate limits.

        Args:
            texts: List of texts to embed
            batch_size: Override default batch size

        Returns:
            List of embedding vectors
        """
        batch_size = batch_size or self.batch_size
        return super().embed_batch(texts, batch_size)

