"""Unit tests for local embedder - focused on embedding behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestLocalEmbedderEmbedding:
    """Tests for actual embedding behavior."""
    
    @patch("biorag.embeddings.local.SentenceTransformer")
    @patch("biorag.embeddings.local.torch")
    def test_embed_documents_returns_vectors(
        self,
        mock_torch: MagicMock,
        mock_st_class: MagicMock,
    ) -> None:
        """embed_documents should return list of float vectors."""
        from biorag.embeddings.local import LocalEmbedder
        
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3] + [0.0] * 381,
            [0.4, 0.5, 0.6] + [0.0] * 381,
        ])
        mock_st_class.return_value = mock_model
        
        embedder = LocalEmbedder()
        embeddings = embedder.embed_documents(["doc1", "doc2"])
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        assert isinstance(embeddings[0][0], float)

    @patch("biorag.embeddings.local.SentenceTransformer")
    @patch("biorag.embeddings.local.torch")
    def test_embed_documents_empty_returns_empty(
        self,
        mock_torch: MagicMock,
        mock_st_class: MagicMock,
    ) -> None:
        """Empty input should return empty list without calling model."""
        from biorag.embeddings.local import LocalEmbedder
        
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model
        
        embedder = LocalEmbedder()
        embeddings = embedder.embed_documents([])
        
        assert embeddings == []
        mock_model.encode.assert_not_called()

    @patch("biorag.embeddings.local.SentenceTransformer")
    @patch("biorag.embeddings.local.torch")
    def test_embed_query_returns_single_vector(
        self,
        mock_torch: MagicMock,
        mock_st_class: MagicMock,
    ) -> None:
        """embed_query should return a single vector."""
        from biorag.embeddings.local import LocalEmbedder
        
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([0.5] * 384)
        mock_st_class.return_value = mock_model
        
        embedder = LocalEmbedder()
        embedding = embedder.embed_query("test query")
        
        assert len(embedding) == 384
        assert isinstance(embedding[0], float)

    @patch("biorag.embeddings.local.SentenceTransformer")
    @patch("biorag.embeddings.local.torch")
    def test_normalization_is_applied_by_default(
        self,
        mock_torch: MagicMock,
        mock_st_class: MagicMock,
    ) -> None:
        """Default should normalize embeddings for cosine similarity."""
        from biorag.embeddings.local import LocalEmbedder
        
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[0.1] * 384])
        mock_st_class.return_value = mock_model
        
        embedder = LocalEmbedder()
        embedder.embed_documents(["test"])
        
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs["normalize_embeddings"] is True

    @patch("biorag.embeddings.local.SentenceTransformer")
    @patch("biorag.embeddings.local.torch")
    def test_batch_size_is_passed_to_model(
        self,
        mock_torch: MagicMock,
        mock_st_class: MagicMock,
    ) -> None:
        """Custom batch_size should be passed to encoder."""
        from biorag.embeddings.local import LocalEmbedder
        
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[0.1] * 384])
        mock_st_class.return_value = mock_model
        
        embedder = LocalEmbedder(batch_size=64)
        embedder.embed_documents(["test"])
        
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs["batch_size"] == 64
