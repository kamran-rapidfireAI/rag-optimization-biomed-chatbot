"""Unit tests for retriever - focused on search modes and result building."""

from __future__ import annotations

import pytest

from biorag.schemas.corpus import Chunk
from biorag.schemas.evaluation import RetrievalResult


class MockFAISSStore:
    """Mock store that returns predictable results for testing."""
    
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # Mock embedder for MMR mode
        self.embedder = type("MockEmbedder", (), {
            "embed_query": lambda self, x: [0.1] * 10,
            "embed_documents": lambda self, x: [[0.1] * 10 for _ in x],
        })()
    
    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        """Return chunks with decreasing scores."""
        return [(c, 1.0 - i * 0.1) for i, c in enumerate(self.chunks[:k])]
    
    def search_batch(self, queries: list[str], k: int = 10) -> list[list[tuple[Chunk, float]]]:
        return [self.search(q, k) for q in queries]


class TestRetrieverSearchModes:
    """Tests for different search modes."""
    
    @pytest.fixture
    def chunks(self) -> list[Chunk]:
        return [
            Chunk(chunk_id=f"doc{i}_0", pmid=f"doc{i}", text=f"Content about topic {i}")
            for i in range(10)
        ]

    def test_similarity_mode_returns_top_k(self, chunks: list[Chunk]) -> None:
        """Similarity mode should return exactly k results."""
        from biorag.retrieve.retriever import Retriever
        
        store = MockFAISSStore(chunks)
        retriever = Retriever(store=store, mode="similarity", k=5)
        
        results = retriever.retrieve("test query")
        
        assert len(results) == 5
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].rank == 1
        assert results[4].rank == 5

    def test_mmr_mode_returns_diverse_results(self, chunks: list[Chunk]) -> None:
        """MMR should return results (diversity tested implicitly by algorithm)."""
        from biorag.retrieve.retriever import Retriever
        
        store = MockFAISSStore(chunks)
        retriever = Retriever(store=store, mode="mmr", k=5, fetch_k=10, lambda_mult=0.5)
        
        results = retriever.retrieve("query")
        
        assert len(results) <= 5
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_threshold_mode_filters_low_scores(self, chunks: list[Chunk]) -> None:
        """Threshold mode should only return results above threshold."""
        from biorag.retrieve.retriever import Retriever
        
        store = MockFAISSStore(chunks)
        # Scores are 1.0, 0.9, 0.8, ... so threshold 0.5 should filter some
        retriever = Retriever(
            store=store,
            mode="similarity_score_threshold",
            k=10,
            fetch_k=10,
            score_threshold=0.5,
        )
        
        results = retriever.retrieve("query")
        
        assert all(r.score >= 0.5 for r in results)

    def test_invalid_mode_raises_error(self, chunks: list[Chunk]) -> None:
        """Invalid mode should raise clear error."""
        from biorag.retrieve.retriever import Retriever
        
        store = MockFAISSStore(chunks)
        retriever = Retriever(store=store, mode="not_a_mode")  # type: ignore
        
        with pytest.raises(ValueError, match="Unknown mode"):
            retriever.retrieve("query")


class TestRetrieverResultBuilding:
    """Tests for how results are constructed."""
    
    def test_results_contain_chunk_data(self) -> None:
        """Results should contain all chunk information."""
        from biorag.retrieve.retriever import Retriever
        
        chunks = [Chunk(chunk_id="abc_0", pmid="abc", text="Specific content here")]
        store = MockFAISSStore(chunks)
        retriever = Retriever(store=store, mode="similarity", k=1)
        
        results = retriever.retrieve("query")
        
        assert results[0].pmid == "abc"
        assert results[0].chunk_id == "abc_0"
        assert results[0].text == "Specific content here"
        assert results[0].score == 1.0  # First result gets score 1.0 from mock

    def test_get_retrieved_pmids_deduplicates(self) -> None:
        """get_retrieved_pmids should return unique PMIDs in order."""
        from biorag.retrieve.retriever import Retriever
        
        # Multiple chunks from same document
        chunks = [
            Chunk(chunk_id="A_0", pmid="A", text="First from A"),
            Chunk(chunk_id="A_1", pmid="A", text="Second from A"),
            Chunk(chunk_id="B_0", pmid="B", text="First from B"),
            Chunk(chunk_id="A_2", pmid="A", text="Third from A"),
        ]
        store = MockFAISSStore(chunks)
        retriever = Retriever(store=store, mode="similarity", k=4)
        
        results = retriever.retrieve("query")
        pmids = retriever.get_retrieved_pmids(results)
        
        # Should be unique and in order of first appearance
        assert pmids == ["A", "B"]


class TestRetrieverBatch:
    """Tests for batch retrieval."""
    
    def test_batch_returns_results_per_query(self) -> None:
        """Batch should return separate result list for each query."""
        from biorag.retrieve.retriever import Retriever
        
        chunks = [Chunk(chunk_id=f"c{i}", pmid=str(i), text=f"Text {i}") for i in range(5)]
        store = MockFAISSStore(chunks)
        retriever = Retriever(store=store, mode="similarity", k=3)
        
        results = retriever.retrieve_batch(["query1", "query2", "query3"])
        
        assert len(results) == 3
        assert all(len(r) == 3 for r in results)


class TestRetrieverEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_store_returns_empty(self) -> None:
        """Empty store should return empty, not error."""
        from biorag.retrieve.retriever import Retriever
        
        store = MockFAISSStore([])
        retriever = Retriever(store=store, mode="similarity", k=10)
        
        results = retriever.retrieve("query")
        
        assert results == []

    def test_k_larger_than_available(self) -> None:
        """k > available chunks should return all available."""
        from biorag.retrieve.retriever import Retriever
        
        chunks = [Chunk(chunk_id="only_0", pmid="only", text="Single chunk")]
        store = MockFAISSStore(chunks)
        retriever = Retriever(store=store, mode="similarity", k=100)
        
        results = retriever.retrieve("query")
        
        assert len(results) == 1
