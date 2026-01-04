"""FAISS vector store for BioRAG Bench."""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from pathlib import Path

import faiss
import numpy as np

from biorag.embeddings.base import Embedder
from biorag.schemas.corpus import Chunk
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


class FAISSStore:
    """FAISS-based vector store for chunk embeddings."""

    def __init__(
        self,
        embedder: Embedder,
        index_type: str = "Flat",
        metric: str = "cosine",
    ) -> None:
        """
        Initialize FAISS store.

        Args:
            embedder: Embedder instance for generating vectors
            index_type: FAISS index type ("Flat", "IVF", "HNSW")
            metric: Distance metric ("cosine", "l2", "ip")
        """
        self.embedder = embedder
        self.index_type = index_type
        self.metric = metric

        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []
        self._id_to_idx: dict[str, int] = {}

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.embedder.embedding_dimension

    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index."""
        d = self.dimension

        if self.index_type == "Flat":
            if self.metric == "cosine" or self.metric == "ip":
                index = faiss.IndexFlatIP(d)
            else:
                index = faiss.IndexFlatL2(d)
        elif self.index_type == "IVF":
            # IVF index for larger datasets
            nlist = 100  # Number of clusters
            if self.metric == "cosine" or self.metric == "ip":
                quantizer = faiss.IndexFlatIP(d)
                index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, nlist)
        elif self.index_type == "HNSW":
            # HNSW for fast approximate search
            m = 32  # Number of connections per layer
            index = faiss.IndexHNSWFlat(d, m)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        return index

    def add_chunks(
        self,
        chunks: list[Chunk] | Iterator[Chunk],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> int:
        """
        Add chunks to the index.

        Args:
            chunks: Chunks to add
            batch_size: Batch size for embedding
            show_progress: Whether to show progress

        Returns:
            Number of chunks added
        """
        chunks_list = list(chunks)
        if not chunks_list:
            return 0

        logger.info(f"Adding {len(chunks_list)} chunks to FAISS index")

        # Initialize index if needed
        if self._index is None:
            self._index = self._create_index()

        # Embed in batches
        all_embeddings = []
        for i in range(0, len(chunks_list), batch_size):
            batch = chunks_list[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.embedder.embed_documents(texts)
            all_embeddings.extend(embeddings)

            if show_progress and (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"Embedded {i + batch_size}/{len(chunks_list)} chunks")

        # Convert to numpy
        vectors = np.array(all_embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(vectors)

        # Train index if needed (for IVF)
        if self.index_type == "IVF" and not self._index.is_trained:
            logger.info("Training IVF index...")
            self._index.train(vectors)

        # Add to index
        start_idx = len(self._chunks)
        self._index.add(vectors)

        # Store chunks and build ID mapping
        for i, chunk in enumerate(chunks_list):
            self._id_to_idx[chunk.chunk_id] = start_idx + i
            self._chunks.append(chunk)

        logger.info(f"Index now contains {self._index.ntotal} vectors")
        return len(chunks_list)

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for similar chunks.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of (chunk, score) tuples
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        # Embed query
        query_vector = np.array([self.embedder.embed_query(query)], dtype=np.float32)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_vector)

        # Search
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._chunks):
                results.append((self._chunks[idx], float(score)))

        return results

    def search_batch(
        self,
        queries: list[str],
        k: int = 10,
    ) -> list[list[tuple[Chunk, float]]]:
        """
        Search for multiple queries in batch.

        Args:
            queries: List of query texts
            k: Number of results per query

        Returns:
            List of result lists
        """
        if self._index is None or self._index.ntotal == 0:
            return [[] for _ in queries]

        # Embed queries
        query_vectors = np.array(
            self.embedder.embed_documents(queries), dtype=np.float32
        )

        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_vectors)

        # Search
        k = min(k, self._index.ntotal)
        all_scores, all_indices = self._index.search(query_vectors, k)

        # Build results
        all_results = []
        for scores, indices in zip(all_scores, all_indices):
            results = []
            for score, idx in zip(scores, indices):
                if idx >= 0 and idx < len(self._chunks):
                    results.append((self._chunks[idx], float(score)))
            all_results.append(results)

        return all_results

    def save(self, path: Path | str) -> None:
        """
        Save index and chunks to disk.

        Args:
            path: Directory to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self._index is not None:
            faiss.write_index(self._index, str(path / "index.faiss"))

        # Save chunks
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

        # Save metadata
        metadata = {
            "index_type": self.index_type,
            "metric": self.metric,
            "dimension": self.dimension,
            "num_chunks": len(self._chunks),
        }
        with open(path / "metadata.json", "w") as f:
            import json

            json.dump(metadata, f, indent=2)

        logger.info(f"Saved FAISS index to {path}")

    @classmethod
    def load(cls, path: Path | str, embedder: Embedder) -> FAISSStore:
        """
        Load index and chunks from disk.

        Args:
            path: Directory to load from
            embedder: Embedder instance

        Returns:
            Loaded FAISSStore
        """
        import json

        path = Path(path)

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        # Create store
        store = cls(
            embedder=embedder,
            index_type=metadata["index_type"],
            metric=metadata["metric"],
        )

        # Load FAISS index
        store._index = faiss.read_index(str(path / "index.faiss"))

        # Load chunks
        with open(path / "chunks.pkl", "rb") as f:
            store._chunks = pickle.load(f)

        # Rebuild ID mapping
        store._id_to_idx = {c.chunk_id: i for i, c in enumerate(store._chunks)}

        logger.info(f"Loaded FAISS index from {path} ({store._index.ntotal} vectors)")
        return store

    @property
    def num_chunks(self) -> int:
        """Return number of indexed chunks."""
        return len(self._chunks)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Get a chunk by ID."""
        idx = self._id_to_idx.get(chunk_id)
        if idx is not None:
            return self._chunks[idx]
        return None

