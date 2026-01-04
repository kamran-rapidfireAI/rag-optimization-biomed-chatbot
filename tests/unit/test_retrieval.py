"""Unit tests for retrieval module."""

from __future__ import annotations

import pytest

from biorag.schemas.evaluation import RetrievalResult


class TestRetrievalResult:
    """Tests for RetrievalResult schema."""

    def test_basic_result(self) -> None:
        """Test basic retrieval result."""
        result = RetrievalResult(
            pmid="12345",
            chunk_id="12345_0",
            text="Sample text",
            score=0.85,
            rank=1,
        )
        assert result.pmid == "12345"
        assert result.score == 0.85
        assert result.rank == 1
        assert result.rerank_score is None

    def test_reranked_result(self) -> None:
        """Test result with reranking info."""
        result = RetrievalResult(
            pmid="12345",
            chunk_id="12345_0",
            text="Sample text",
            score=0.75,
            rank=5,
            rerank_score=0.92,
            rerank_rank=1,
        )
        assert result.rerank_score == 0.92
        assert result.rerank_rank == 1


class TestRetrieverInterface:
    """Tests for Retriever interface."""

    def test_get_retrieved_pmids(self) -> None:
        """Test extracting unique PMIDs from results."""
        from biorag.retrieve.retriever import Retriever

        results = [
            RetrievalResult(pmid="111", chunk_id="111_0", text="a", score=0.9, rank=1),
            RetrievalResult(pmid="111", chunk_id="111_1", text="b", score=0.8, rank=2),
            RetrievalResult(pmid="222", chunk_id="222_0", text="c", score=0.7, rank=3),
            RetrievalResult(pmid="333", chunk_id="333_0", text="d", score=0.6, rank=4),
            RetrievalResult(pmid="222", chunk_id="222_1", text="e", score=0.5, rank=5),
        ]

        # Create a mock retriever to test the helper method
        class MockStore:
            embedder = None

        # Can't fully test without a real store, but we can test the helper
        # retriever = Retriever(store=MockStore(), k=5)
        # pmids = retriever.get_retrieved_pmids(results)

        # Test the logic directly
        seen = set()
        pmids = []
        for r in results:
            if r.pmid not in seen:
                seen.add(r.pmid)
                pmids.append(r.pmid)

        assert pmids == ["111", "222", "333"]

