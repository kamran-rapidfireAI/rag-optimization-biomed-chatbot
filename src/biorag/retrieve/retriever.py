"""Retriever module for BioRAG Bench."""

from __future__ import annotations

import time
from typing import Literal

from biorag.indexing.faiss_store import FAISSStore
from biorag.schemas.corpus import Chunk
from biorag.schemas.evaluation import RetrievalResult
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Configurable retriever with multiple search modes.

    Supports:
    - similarity: Standard top-k similarity search
    - mmr: Maximal Marginal Relevance for diversity
    - similarity_score_threshold: Filter by minimum score
    """

    def __init__(
        self,
        store: FAISSStore,
        mode: Literal["similarity", "mmr", "similarity_score_threshold"] = "mmr",
        k: int = 10,
        fetch_k: int = 50,
        lambda_mult: float = 0.5,
        score_threshold: float | None = None,
    ) -> None:
        """
        Initialize retriever.

        Args:
            store: FAISSStore instance
            mode: Search mode
            k: Number of results to return
            fetch_k: Number of candidates to fetch for MMR
            lambda_mult: MMR diversity parameter (0=max diversity, 1=max relevance)
            score_threshold: Minimum score for threshold mode
        """
        self.store = store
        self.mode = mode
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query text

        Returns:
            List of RetrievalResult objects
        """
        start_time = time.perf_counter()

        if self.mode == "similarity":
            results = self._similarity_search(query)
        elif self.mode == "mmr":
            results = self._mmr_search(query)
        elif self.mode == "similarity_score_threshold":
            results = self._threshold_search(query)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.log_retrieval(query, len(results), latency_ms)

        return results

    def retrieve_batch(self, queries: list[str]) -> list[list[RetrievalResult]]:
        """
        Retrieve for multiple queries efficiently.

        Args:
            queries: List of query texts

        Returns:
            List of result lists
        """
        if self.mode == "similarity":
            # Use batch search for efficiency
            raw_results = self.store.search_batch(queries, k=self.k)
            return [
                self._build_results(results)
                for results in raw_results
            ]
        else:
            # Fall back to individual searches for MMR/threshold
            return [self.retrieve(q) for q in queries]

    def _similarity_search(self, query: str) -> list[RetrievalResult]:
        """Standard similarity search."""
        results = self.store.search(query, k=self.k)
        return self._build_results(results)

    def _mmr_search(self, query: str) -> list[RetrievalResult]:
        """
        Maximal Marginal Relevance search for diversity.

        MMR iteratively selects documents that are:
        - Relevant to the query
        - Different from already selected documents
        """
        import numpy as np

        # Fetch more candidates than needed
        candidates = self.store.search(query, k=self.fetch_k)

        if not candidates:
            return []

        # Get query embedding
        query_embedding = np.array(self.store.embedder.embed_query(query))

        # Get candidate embeddings
        candidate_texts = [c.text for c, _ in candidates]
        candidate_embeddings = np.array(
            self.store.embedder.embed_documents(candidate_texts)
        )

        # Initialize selected set
        selected_indices: list[int] = []
        remaining_indices = list(range(len(candidates)))

        # MMR selection loop
        while len(selected_indices) < self.k and remaining_indices:
            mmr_scores = []

            for idx in remaining_indices:
                # Similarity to query
                query_sim = float(np.dot(query_embedding, candidate_embeddings[idx]))

                # Max similarity to already selected
                if selected_indices:
                    selected_embeddings = candidate_embeddings[selected_indices]
                    doc_sims = np.dot(selected_embeddings, candidate_embeddings[idx])
                    max_doc_sim = float(np.max(doc_sims))
                else:
                    max_doc_sim = 0.0

                # MMR score
                mmr = self.lambda_mult * query_sim - (1 - self.lambda_mult) * max_doc_sim
                mmr_scores.append((idx, mmr))

            # Select highest MMR score
            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Build results from selected indices
        selected_candidates = [(candidates[i][0], candidates[i][1]) for i in selected_indices]
        return self._build_results(selected_candidates)

    def _threshold_search(self, query: str) -> list[RetrievalResult]:
        """Search with score threshold filtering."""
        results = self.store.search(query, k=self.fetch_k)

        if self.score_threshold is not None:
            results = [(c, s) for c, s in results if s >= self.score_threshold]

        # Take top k after filtering
        results = results[: self.k]
        return self._build_results(results)

    def _build_results(
        self, results: list[tuple[Chunk, float]]
    ) -> list[RetrievalResult]:
        """Convert raw results to RetrievalResult objects."""
        return [
            RetrievalResult(
                pmid=chunk.pmid,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=score,
                rank=i + 1,
            )
            for i, (chunk, score) in enumerate(results)
        ]

    def get_retrieved_pmids(self, results: list[RetrievalResult]) -> list[str]:
        """Extract unique PMIDs from results in order."""
        seen = set()
        pmids = []
        for r in results:
            if r.pmid not in seen:
                seen.add(r.pmid)
                pmids.append(r.pmid)
        return pmids

