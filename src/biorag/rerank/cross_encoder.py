"""Cross-encoder reranker for BioRAG Bench."""

from __future__ import annotations

import time

import torch
from sentence_transformers import CrossEncoder

from biorag.rerank.base import Reranker
from biorag.schemas.evaluation import RetrievalResult
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder reranker using sentence-transformers.

    Uses GPU acceleration when available for efficient batch processing.
    """

    # Recommended models
    MODELS = {
        "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "tiny": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "balanced": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "biomedical": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    }

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 50,
        final_k: int = 8,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        """
        Initialize cross-encoder reranker.

        Args:
            model: HuggingFace model name or path
            top_n: Number of candidates to rerank
            final_k: Number of results to return after reranking
            batch_size: Batch size for scoring
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        super().__init__(model=model, top_n=top_n, final_k=final_k)

        self.batch_size = batch_size

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Loading cross-encoder model: {model} on {device}")
        self._model = CrossEncoder(model, device=device)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Rerank results using cross-encoder.

        Args:
            query: Original query
            results: Retrieval results to rerank

        Returns:
            Reranked results with updated scores and ranks
        """
        if not results:
            return []

        start_time = time.perf_counter()

        # Limit to top_n candidates
        candidates = results[: self.top_n]

        # Prepare query-document pairs
        pairs = [(query, r.text) for r in candidates]

        # Score with cross-encoder
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Combine with original results and sort
        scored_results = list(zip(candidates, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Build reranked results
        reranked = []
        for new_rank, (result, score) in enumerate(scored_results[: self.final_k]):
            # Create new result with rerank info
            reranked.append(
                RetrievalResult(
                    pmid=result.pmid,
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=result.score,
                    rank=result.rank,
                    rerank_score=float(score),
                    rerank_rank=new_rank + 1,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.log_rerank(
            input_count=len(candidates),
            output_count=len(reranked),
            latency_ms=latency_ms,
            model=self.model,
        )

        return reranked

    def rerank_batch(
        self,
        queries: list[str],
        results_batch: list[list[RetrievalResult]],
    ) -> list[list[RetrievalResult]]:
        """
        Rerank multiple queries with batched scoring.

        Args:
            queries: List of queries
            results_batch: List of result lists

        Returns:
            List of reranked result lists
        """
        if not queries:
            return []

        start_time = time.perf_counter()

        # Prepare all pairs
        all_pairs = []
        pair_indices = []  # Track which query each pair belongs to

        for query_idx, (query, results) in enumerate(zip(queries, results_batch)):
            candidates = results[: self.top_n]
            for result in candidates:
                all_pairs.append((query, result.text))
                pair_indices.append((query_idx, result))

        if not all_pairs:
            return [[] for _ in queries]

        # Score all pairs at once
        all_scores = self._model.predict(
            all_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Group scores by query
        query_results: dict[int, list[tuple[RetrievalResult, float]]] = {
            i: [] for i in range(len(queries))
        }

        for (query_idx, result), score in zip(pair_indices, all_scores):
            query_results[query_idx].append((result, float(score)))

        # Sort and build final results
        all_reranked = []
        for query_idx in range(len(queries)):
            scored = query_results[query_idx]
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for new_rank, (result, score) in enumerate(scored[: self.final_k]):
                reranked.append(
                    RetrievalResult(
                        pmid=result.pmid,
                        chunk_id=result.chunk_id,
                        text=result.text,
                        score=result.score,
                        rank=result.rank,
                        rerank_score=score,
                        rerank_rank=new_rank + 1,
                    )
                )
            all_reranked.append(reranked)

        latency_ms = (time.perf_counter() - start_time) * 1000
        total_input = sum(len(r[: self.top_n]) for r in results_batch)
        total_output = sum(len(r) for r in all_reranked)
        logger.log_rerank(
            input_count=total_input,
            output_count=total_output,
            latency_ms=latency_ms,
            model=self.model,
        )

        return all_reranked

