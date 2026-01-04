"""Local HuggingFace embeddings for BioRAG Bench."""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

from biorag.embeddings.base import Embedder
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


class LocalEmbedder(Embedder):
    """Local embeddings using sentence-transformers models."""

    # Recommended models for biomedical text
    BIOMEDICAL_MODELS = [
        "pritamdeka/S-PubMedBert-MS-MARCO",
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "allenai/specter2",
    ]

    # General-purpose fast models
    GENERAL_MODELS = [
        "all-MiniLM-L6-v2",  # Fast and light
        "all-mpnet-base-v2",  # Better quality
        "BAAI/bge-small-en-v1.5",  # Good balance
    ]

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Initialize local embedder.

        Args:
            model: HuggingFace model name or path
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to L2-normalize embeddings
        """
        super().__init__(model=model)

        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Loading embedding model: {model} on {device}")
        self._model = SentenceTransformer(model, device=device)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents using local model.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query using local model.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        embedding = self._model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        return embedding.tolist()

