"""Unit tests for cross-encoder reranker - focused on actual reranking behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from biorag.schemas.evaluation import RetrievalResult


class TestCrossEncoderReranking:
    """Tests for the core reranking behavior."""
    
    @pytest.fixture
    def sample_results(self) -> list[RetrievalResult]:
        """Create retrieval results in original ranking order."""
        return [
            RetrievalResult(pmid="1", chunk_id="1_0", text="First in original ranking", score=0.95, rank=1),
            RetrievalResult(pmid="2", chunk_id="2_0", text="Second - will be best after rerank", score=0.90, rank=2),
            RetrievalResult(pmid="3", chunk_id="3_0", text="Third in original", score=0.85, rank=3),
            RetrievalResult(pmid="4", chunk_id="4_0", text="Fourth - will be second after rerank", score=0.80, rank=4),
            RetrievalResult(pmid="5", chunk_id="5_0", text="Fifth in original", score=0.75, rank=5),
        ]

    @patch("biorag.rerank.cross_encoder.CrossEncoder")
    @patch("biorag.rerank.cross_encoder.torch")
    def test_reranking_reorders_by_cross_encoder_scores(
        self,
        mock_torch: MagicMock,
        mock_ce_class: MagicMock,
        sample_results: list[RetrievalResult],
    ) -> None:
        """Reranker should reorder results based on cross-encoder scores, not original scores."""
        from biorag.rerank.cross_encoder import CrossEncoderReranker
        
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        # Cross-encoder scores differ from original ranking
        # chunk_id "2_0" gets highest score (0.9), "4_0" gets second (0.7)
        mock_model.predict.return_value = np.array([0.3, 0.9, 0.5, 0.7, 0.1])
        mock_ce_class.return_value = mock_model
        
        reranker = CrossEncoderReranker(model="test", top_n=10, final_k=3)
        reranked = reranker.rerank("test query", sample_results)
        
        # Top result should be chunk_id "2_0" (score 0.9), not "1_0"
        assert reranked[0].chunk_id == "2_0"
        assert reranked[0].rerank_score == 0.9
        assert reranked[0].rerank_rank == 1
        
        # Second should be "4_0" (score 0.7)
        assert reranked[1].chunk_id == "4_0"
        assert reranked[1].rerank_score == 0.7
        
        # Results are sorted by rerank_score descending
        scores = [r.rerank_score for r in reranked]
        assert scores == sorted(scores, reverse=True)

    @patch("biorag.rerank.cross_encoder.CrossEncoder")
    @patch("biorag.rerank.cross_encoder.torch")
    def test_original_scores_and_ranks_preserved(
        self,
        mock_torch: MagicMock,
        mock_ce_class: MagicMock,
        sample_results: list[RetrievalResult],
    ) -> None:
        """Reranked results should preserve original retrieval info for analysis."""
        from biorag.rerank.cross_encoder import CrossEncoderReranker
        
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.9, 0.3, 0.7, 0.1])
        mock_ce_class.return_value = mock_model
        
        reranker = CrossEncoderReranker(model="test", final_k=5)
        reranked = reranker.rerank("query", sample_results)
        
        for r in reranked:
            orig = next(o for o in sample_results if o.chunk_id == r.chunk_id)
            assert r.score == orig.score  # Original retrieval score
            assert r.rank == orig.rank    # Original rank
            assert r.text == orig.text    # Original content

    @patch("biorag.rerank.cross_encoder.CrossEncoder")
    @patch("biorag.rerank.cross_encoder.torch")
    def test_top_n_limits_candidates_scored(
        self,
        mock_torch: MagicMock,
        mock_ce_class: MagicMock,
    ) -> None:
        """Only top_n candidates should be scored by cross-encoder (efficiency)."""
        from biorag.rerank.cross_encoder import CrossEncoderReranker
        
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.5, 0.7])  # Only 3 scores
        mock_ce_class.return_value = mock_model
        
        reranker = CrossEncoderReranker(model="test", top_n=3, final_k=2)
        
        # Give 10 results, but only top 3 should be scored
        results = [
            RetrievalResult(pmid=str(i), chunk_id=f"{i}_0", text=f"Chunk {i}", score=0.9-i*0.05, rank=i+1)
            for i in range(10)
        ]
        
        reranker.rerank("query", results)
        
        # Cross-encoder should receive exactly top_n pairs
        pairs = mock_model.predict.call_args[0][0]
        assert len(pairs) == 3

    @patch("biorag.rerank.cross_encoder.CrossEncoder")
    @patch("biorag.rerank.cross_encoder.torch")
    def test_empty_input_returns_empty(
        self,
        mock_torch: MagicMock,
        mock_ce_class: MagicMock,
    ) -> None:
        """Empty input should return empty without calling model."""
        from biorag.rerank.cross_encoder import CrossEncoderReranker
        
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_ce_class.return_value = mock_model
        
        reranker = CrossEncoderReranker(model="test")
        result = reranker.rerank("query", [])
        
        assert result == []
        mock_model.predict.assert_not_called()


class TestCrossEncoderBatchProcessing:
    """Tests for batch reranking (multiple queries at once)."""
    
    @patch("biorag.rerank.cross_encoder.CrossEncoder")
    @patch("biorag.rerank.cross_encoder.torch")
    def test_batch_rerank_scores_all_queries_together(
        self,
        mock_torch: MagicMock,
        mock_ce_class: MagicMock,
    ) -> None:
        """Batch reranking should process all query-doc pairs in one model call."""
        from biorag.rerank.cross_encoder import CrossEncoderReranker
        
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        # 2 queries x 2 results = 4 pairs
        mock_model.predict.return_value = np.array([0.9, 0.3, 0.5, 0.8])
        mock_ce_class.return_value = mock_model
        
        reranker = CrossEncoderReranker(model="test", top_n=10, final_k=2)
        
        results_batch = [
            [RetrievalResult(pmid="1", chunk_id="1_0", text="Q1 doc1", score=0.9, rank=1),
             RetrievalResult(pmid="2", chunk_id="2_0", text="Q1 doc2", score=0.8, rank=2)],
            [RetrievalResult(pmid="3", chunk_id="3_0", text="Q2 doc1", score=0.85, rank=1),
             RetrievalResult(pmid="4", chunk_id="4_0", text="Q2 doc2", score=0.75, rank=2)],
        ]
        
        reranked = reranker.rerank_batch(["query1", "query2"], results_batch)
        
        # Should return results for both queries
        assert len(reranked) == 2
        # Single model.predict call for efficiency
        assert mock_model.predict.call_count == 1

    @patch("biorag.rerank.cross_encoder.CrossEncoder")
    @patch("biorag.rerank.cross_encoder.torch")
    def test_batch_handles_empty_result_lists(
        self,
        mock_torch: MagicMock,
        mock_ce_class: MagicMock,
    ) -> None:
        """Batch with some empty result lists should not fail."""
        from biorag.rerank.cross_encoder import CrossEncoderReranker
        
        mock_torch.cuda.is_available.return_value = False
        mock_ce_class.return_value = MagicMock()
        
        reranker = CrossEncoderReranker(model="test")
        reranked = reranker.rerank_batch(["q1", "q2"], [[], []])
        
        assert reranked == [[], []]
