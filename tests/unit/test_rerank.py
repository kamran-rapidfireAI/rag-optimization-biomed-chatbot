"""Unit tests for reranking module."""

from __future__ import annotations

import pytest

from biorag.rerank.base import Reranker
from biorag.schemas.evaluation import RetrievalResult


class TestRerankerInterface:
    """Tests for Reranker interface."""

    def test_abstract_reranker(self) -> None:
        """Test that Reranker is abstract."""
        with pytest.raises(TypeError):
            Reranker(model="test")  # type: ignore

    def test_concrete_implementation(self) -> None:
        """Test implementing the interface."""

        class MockReranker(Reranker):
            def rerank(
                self, query: str, results: list[RetrievalResult]
            ) -> list[RetrievalResult]:
                # Simple mock: reverse the order
                reranked = []
                for i, r in enumerate(reversed(results[: self.final_k])):
                    reranked.append(
                        RetrievalResult(
                            pmid=r.pmid,
                            chunk_id=r.chunk_id,
                            text=r.text,
                            score=r.score,
                            rank=r.rank,
                            rerank_score=1.0 - (i * 0.1),
                            rerank_rank=i + 1,
                        )
                    )
                return reranked

        reranker = MockReranker(model="mock", top_n=10, final_k=5)
        assert reranker.model == "mock"
        assert reranker.top_n == 10
        assert reranker.final_k == 5

        # Test reranking
        results = [
            RetrievalResult(
                pmid=str(i), chunk_id=f"{i}_0", text=f"text{i}", score=0.9 - i * 0.1, rank=i + 1
            )
            for i in range(10)
        ]

        reranked = reranker.rerank("query", results)
        assert len(reranked) == 5
        assert all(r.rerank_score is not None for r in reranked)
        assert all(r.rerank_rank is not None for r in reranked)

    def test_batch_rerank_default(self) -> None:
        """Test default batch implementation."""

        class SimpleReranker(Reranker):
            def rerank(
                self, query: str, results: list[RetrievalResult]
            ) -> list[RetrievalResult]:
                # Just return first final_k with rerank info
                return [
                    RetrievalResult(
                        pmid=r.pmid,
                        chunk_id=r.chunk_id,
                        text=r.text,
                        score=r.score,
                        rank=r.rank,
                        rerank_score=r.score,
                        rerank_rank=i + 1,
                    )
                    for i, r in enumerate(results[: self.final_k])
                ]

        reranker = SimpleReranker(model="simple", final_k=3)

        queries = ["query1", "query2"]
        results_batch = [
            [
                RetrievalResult(
                    pmid=f"q1_{i}", chunk_id=f"q1_{i}_0", text="a", score=0.9, rank=i + 1
                )
                for i in range(5)
            ],
            [
                RetrievalResult(
                    pmid=f"q2_{i}", chunk_id=f"q2_{i}_0", text="b", score=0.8, rank=i + 1
                )
                for i in range(5)
            ],
        ]

        reranked_batch = reranker.rerank_batch(queries, results_batch)

        assert len(reranked_batch) == 2
        assert len(reranked_batch[0]) == 3
        assert len(reranked_batch[1]) == 3

