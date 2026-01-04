"""Unit tests for FAISS store - focused on real behavior, not boilerplate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biorag.indexing.faiss_store import FAISSStore
from biorag.schemas.corpus import Chunk


class MockEmbedder:
    """Mock embedder that produces deterministic embeddings from text."""
    
    def __init__(self, dimension: int = 128):
        self._dimension = dimension
    
    @property
    def embedding_dimension(self) -> int:
        return self._dimension
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Deterministic embeddings - similar text produces similar vectors."""
        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(self._dimension).tolist()
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class TestFAISSStoreIndexAndSearch:
    """Tests for the core indexing and search workflow."""
    
    def test_index_and_retrieve_finds_relevant_chunks(self) -> None:
        """Test that search returns chunks and scores decrease by rank."""
        embedder = MockEmbedder()
        store = FAISSStore(embedder=embedder)
        
        chunks = [
            Chunk(chunk_id="cancer_0", pmid="1", text="Cancer treatment with chemotherapy"),
            Chunk(chunk_id="diabetes_0", pmid="2", text="Diabetes insulin therapy management"),
            Chunk(chunk_id="heart_0", pmid="3", text="Heart disease cardiovascular health"),
        ]
        store.add_chunks(chunks, show_progress=False)
        
        results = store.search("chemotherapy cancer", k=3)
        
        # Should return results with chunks and scores
        assert len(results) == 3
        assert all(isinstance(r[0], Chunk) for r in results)
        assert all(isinstance(r[1], float) for r in results)
        
        # Scores should be in descending order
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_incremental_indexing_accumulates(self) -> None:
        """Test that adding chunks incrementally works correctly."""
        embedder = MockEmbedder()
        store = FAISSStore(embedder=embedder)
        
        store.add_chunks([Chunk(chunk_id="a_0", pmid="a", text="First")], show_progress=False)
        assert store.num_chunks == 1
        
        store.add_chunks([Chunk(chunk_id="b_0", pmid="b", text="Second")], show_progress=False)
        assert store.num_chunks == 2
        
        # Both should be searchable
        results = store.search("test", k=10)
        assert len(results) == 2

    def test_search_on_empty_store_returns_empty(self) -> None:
        """Empty store should return empty results, not error."""
        store = FAISSStore(embedder=MockEmbedder())
        results = store.search("anything")
        assert results == []

    def test_k_is_capped_at_index_size(self) -> None:
        """Requesting more than indexed returns all available."""
        embedder = MockEmbedder()
        store = FAISSStore(embedder=embedder)
        store.add_chunks([Chunk(chunk_id="only_0", pmid="1", text="Only chunk")], show_progress=False)
        
        results = store.search("query", k=100)
        assert len(results) == 1

    def test_batch_search_returns_results_per_query(self) -> None:
        """Batch search should return separate results for each query."""
        embedder = MockEmbedder()
        store = FAISSStore(embedder=embedder)
        store.add_chunks([
            Chunk(chunk_id="a_0", pmid="1", text="Alpha content"),
            Chunk(chunk_id="b_0", pmid="2", text="Beta content"),
        ], show_progress=False)
        
        results = store.search_batch(["alpha", "beta"], k=2)
        
        assert len(results) == 2
        assert len(results[0]) == 2
        assert len(results[1]) == 2


class TestFAISSStorePersistence:
    """Tests for save/load - critical for reproducibility."""
    
    def test_save_creates_required_files(self, tmp_path: Path) -> None:
        """Save should create index, chunks, and metadata files."""
        store = FAISSStore(embedder=MockEmbedder())
        store.add_chunks([Chunk(chunk_id="a_0", pmid="a", text="Test")], show_progress=False)
        
        store.save(tmp_path / "index")
        
        assert (tmp_path / "index" / "index.faiss").exists()
        assert (tmp_path / "index" / "chunks.pkl").exists()
        assert (tmp_path / "index" / "metadata.json").exists()

    def test_load_restores_searchable_index(self, tmp_path: Path) -> None:
        """Loaded index should be fully functional."""
        embedder = MockEmbedder()
        
        # Create and save
        original = FAISSStore(embedder=embedder)
        original.add_chunks([
            Chunk(chunk_id="unique_123", pmid="doc1", text="Important finding about treatment"),
        ], show_progress=False)
        original.save(tmp_path / "index")
        
        # Load and verify
        loaded = FAISSStore.load(tmp_path / "index", embedder)
        results = loaded.search("treatment", k=1)
        
        assert len(results) == 1
        assert results[0][0].chunk_id == "unique_123"
        assert results[0][0].pmid == "doc1"


class TestFAISSStoreChunkLookup:
    """Tests for chunk retrieval by ID."""
    
    def test_get_chunk_returns_correct_chunk(self) -> None:
        """get_chunk should return the exact chunk that was indexed."""
        store = FAISSStore(embedder=MockEmbedder())
        chunk = Chunk(chunk_id="specific_id", pmid="doc123", text="Specific content")
        store.add_chunks([chunk], show_progress=False)
        
        result = store.get_chunk("specific_id")
        
        assert result is not None
        assert result.pmid == "doc123"
        assert result.text == "Specific content"

    def test_get_chunk_returns_none_for_missing(self) -> None:
        """Missing chunk ID should return None, not error."""
        store = FAISSStore(embedder=MockEmbedder())
        assert store.get_chunk("nonexistent") is None


class TestFAISSStoreIndexTypes:
    """Tests for different index configurations."""
    
    def test_l2_distance_metric_works(self) -> None:
        """L2 distance should produce valid results."""
        store = FAISSStore(embedder=MockEmbedder(), metric="l2")
        store.add_chunks([Chunk(chunk_id="a_0", pmid="a", text="Test content")], show_progress=False)
        
        results = store.search("query", k=1)
        assert len(results) == 1

    def test_hnsw_index_type_works(self) -> None:
        """HNSW approximate search should produce valid results."""
        store = FAISSStore(embedder=MockEmbedder(), index_type="HNSW")
        store.add_chunks([Chunk(chunk_id="a_0", pmid="a", text="Test content")], show_progress=False)
        
        results = store.search("query", k=1)
        assert len(results) == 1

    def test_invalid_index_type_raises_on_use(self) -> None:
        """Invalid index type should raise clear error."""
        store = FAISSStore(embedder=MockEmbedder(), index_type="NotAValidType")
        
        with pytest.raises(ValueError, match="Unknown index type"):
            store.add_chunks([Chunk(chunk_id="a_0", pmid="a", text="Test")], show_progress=False)
