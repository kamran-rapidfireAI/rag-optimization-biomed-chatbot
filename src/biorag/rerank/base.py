"""Base reranker interface for BioRAG Bench."""

from __future__ import annotations

from abc import ABC, abstractmethod

from biorag.schemas.evaluation import RetrievalResult


class Reranker(ABC):
    """Abstract base class for rerankers."""

    def __init__(
        self,
        model: str,
        top_n: int = 50,
        final_k: int = 8,
    ) -> None:
        """
        Initialize reranker.

        Args:
            model: Model name/identifier
            top_n: Number of candidates to rerank
            final_k: Number of results to return after reranking
        """
        self.model = model
        self.top_n = top_n
        self.final_k = final_k

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Rerank retrieval results.

        Args:
            query: Original query
            results: Retrieval results to rerank

        Returns:
            Reranked results with updated scores and ranks
        """
        pass

    def rerank_batch(
        self,
        queries: list[str],
        results_batch: list[list[RetrievalResult]],
    ) -> list[list[RetrievalResult]]:
        """
        Rerank multiple queries.

        Args:
            queries: List of queries
            results_batch: List of result lists

        Returns:
            List of reranked result lists
        """
        return [
            self.rerank(query, results)
            for query, results in zip(queries, results_batch)
        ]

