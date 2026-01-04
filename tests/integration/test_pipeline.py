"""Integration tests for the RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from biorag.embeddings.base import Embedder
from biorag.indexing.faiss_store import FAISSStore
from biorag.pipeline.rag import (
    PipelineDebugInfo,
    PipelineLatency,
    RAGPipeline,
    RAGResult,
)
from biorag.schemas.config import BioRAGConfig
from biorag.schemas.corpus import Chunk
from biorag.schemas.evaluation import RetrievalResult
from biorag.schemas.generation import AnswerOutput, GenerationResponse


class MockEmbedder(Embedder):
    """Mock embedder for testing."""

    def __init__(self, dimension: int = 128) -> None:
        super().__init__(model="mock-embedder", dimension=dimension)
        self._mock_dimension = dimension

    @property
    def embedding_dimension(self) -> int:
        return self._mock_dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        return [
            np.random.randn(self._mock_dimension).tolist()
            for _ in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        import numpy as np

        return np.random.randn(self._mock_dimension).tolist()


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    """Create a mock embedder."""
    return MockEmbedder(dimension=128)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(
            pmid="12345678",
            chunk_id="12345678_0",
            text="BRCA1 is a tumor suppressor gene involved in DNA repair.",
            start_char=0,
            end_char=55,
        ),
        Chunk(
            pmid="12345679",
            chunk_id="12345679_0",
            text="Metformin is a medication used to treat type 2 diabetes.",
            start_char=0,
            end_char=55,
        ),
        Chunk(
            pmid="12345680",
            chunk_id="12345680_0",
            text="COVID-19 is caused by the SARS-CoV-2 virus.",
            start_char=0,
            end_char=43,
        ),
        Chunk(
            pmid="12345681",
            chunk_id="12345681_0",
            text="Aspirin is commonly used for pain relief and heart health.",
            start_char=0,
            end_char=57,
        ),
        Chunk(
            pmid="12345682",
            chunk_id="12345682_0",
            text="The p53 protein is a critical tumor suppressor in humans.",
            start_char=0,
            end_char=56,
        ),
    ]


@pytest.fixture
def faiss_store_with_data(
    mock_embedder: MockEmbedder,
    sample_chunks: list[Chunk],
) -> FAISSStore:
    """Create a FAISS store with sample data."""
    store = FAISSStore(embedder=mock_embedder, metric="cosine")
    store.add_chunks(sample_chunks, show_progress=False)
    return store


@pytest.fixture
def mock_generator_response() -> GenerationResponse:
    """Create a mock generation response."""
    return GenerationResponse(
        answer=AnswerOutput(
            answer="BRCA1 is a tumor suppressor gene that plays a crucial role in DNA repair.",
            answer_type="direct",
            supported_by_evidence=True,
        ),
        model="gpt-4o-mini",
        prompt_template="test_template",
        input_tokens=500,
        output_tokens=50,
        latency_ms=200.0,
        cache_hit=False,
    )


class TestRAGPipeline:
    """Tests for RAGPipeline."""

    def test_pipeline_initialization(self) -> None:
        """Test pipeline can be initialized with default config."""
        config = BioRAGConfig()
        pipeline = RAGPipeline(config=config)
        
        assert pipeline.config == config
        assert pipeline._debug_info is not None

    def test_pipeline_debug_info(self) -> None:
        """Test debug info is correctly built from config."""
        config = BioRAGConfig()
        pipeline = RAGPipeline(config=config)
        
        debug_info = pipeline._debug_info
        assert debug_info.chunking["type"] == config.chunking.type
        assert debug_info.retrieval["mode"] == config.retrieval.mode
        assert debug_info.rerank["enabled"] == config.rerank.enabled

    def test_pipeline_requires_index(self) -> None:
        """Test pipeline raises error when FAISS store is not loaded."""
        config = BioRAGConfig()
        pipeline = RAGPipeline(config=config)
        
        with pytest.raises(ValueError, match="FAISS store not initialized"):
            _ = pipeline.faiss_store

    def test_pipeline_with_mock_store(
        self,
        faiss_store_with_data: FAISSStore,
        mock_generator_response: GenerationResponse,
    ) -> None:
        """Test pipeline with mocked components."""
        config = BioRAGConfig(
            rerank={"enabled": False},  # Disable reranking for simpler test
        )
        
        # Create pipeline with injected store
        pipeline = RAGPipeline(
            config=config,
            faiss_store=faiss_store_with_data,
        )
        
        # Mock the generator
        mock_generator = MagicMock()
        mock_generator.generate.return_value = mock_generator_response
        pipeline._generator = mock_generator
        
        # Run pipeline
        result = pipeline.query(
            question="What is BRCA1?",
            question_type="factoid",
        )
        
        # Verify result
        assert isinstance(result, RAGResult)
        assert len(result.retrieved_chunks) > 0
        assert result.answer.answer == mock_generator_response.answer.answer
        assert result.latency.total_ms > 0

    def test_retrieve_only(
        self,
        faiss_store_with_data: FAISSStore,
    ) -> None:
        """Test retrieve_only method."""
        config = BioRAGConfig()
        pipeline = RAGPipeline(
            config=config,
            faiss_store=faiss_store_with_data,
        )
        
        results = pipeline.retrieve_only("What is cancer?")
        
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_skip_generation(
        self,
        faiss_store_with_data: FAISSStore,
    ) -> None:
        """Test query with skip_generation=True."""
        config = BioRAGConfig(rerank={"enabled": False})
        pipeline = RAGPipeline(
            config=config,
            faiss_store=faiss_store_with_data,
        )
        
        result = pipeline.query(
            question="What is cancer?",
            skip_generation=True,
        )
        
        assert result.answer.abstained
        assert result.answer.abstention_reason == "Generation skipped"
        assert len(result.retrieved_chunks) > 0
        assert result.latency.generate_ms == 0.0

    def test_get_config_summary(self) -> None:
        """Test get_config_summary returns expected structure."""
        config = BioRAGConfig()
        pipeline = RAGPipeline(config=config)
        
        summary = pipeline.get_config_summary()
        
        assert "llm" in summary
        assert "embeddings" in summary
        assert "chunking" in summary
        assert "retrieval" in summary
        assert "rerank" in summary
        assert summary["llm"]["model"] == config.llm.model


class TestPipelineLatency:
    """Tests for PipelineLatency."""

    def test_latency_to_dict(self) -> None:
        """Test latency conversion to dict."""
        latency = PipelineLatency(
            retrieve_ms=50.0,
            rerank_ms=100.0,
            generate_ms=200.0,
            total_ms=350.0,
        )
        
        result = latency.to_dict()
        
        assert result["retrieve_ms"] == 50.0
        assert result["rerank_ms"] == 100.0
        assert result["generate_ms"] == 200.0
        assert result["total_ms"] == 350.0


class TestPipelineDebugInfo:
    """Tests for PipelineDebugInfo."""

    def test_debug_info_to_dict(self) -> None:
        """Test debug info conversion to dict."""
        debug_info = PipelineDebugInfo(
            chunking={"type": "recursive", "chunk_size": 350},
            retrieval={"mode": "mmr", "k": 10},
            rerank={"enabled": True, "model": "test-model"},
            generation={"model": "gpt-4o-mini"},
        )
        
        result = debug_info.to_dict()
        
        assert result["chunking"]["type"] == "recursive"
        assert result["retrieval"]["mode"] == "mmr"
        assert result["rerank"]["enabled"]


class TestRAGResult:
    """Tests for RAGResult."""

    def test_result_to_dict(
        self,
        mock_generator_response: GenerationResponse,
    ) -> None:
        """Test RAGResult conversion to dict."""
        result = RAGResult(
            answer=mock_generator_response.answer,
            retrieved_chunks=[
                RetrievalResult(
                    pmid="12345678",
                    chunk_id="12345678_0",
                    text="Sample text",
                    score=0.95,
                    rank=1,
                )
            ],
            reranked_chunks=[
                RetrievalResult(
                    pmid="12345678",
                    chunk_id="12345678_0",
                    text="Sample text",
                    score=0.95,
                    rank=1,
                    rerank_score=0.98,
                    rerank_rank=1,
                )
            ],
            latency=PipelineLatency(
                retrieve_ms=50.0,
                rerank_ms=100.0,
                generate_ms=200.0,
                total_ms=350.0,
            ),
            debug_info=PipelineDebugInfo(),
            generation_response=mock_generator_response,
        )
        
        result_dict = result.to_dict()
        
        assert "answer" in result_dict
        assert "retrieved_chunks" in result_dict
        assert "reranked_chunks" in result_dict
        assert "latency" in result_dict
        assert "debug_info" in result_dict
        assert len(result_dict["retrieved_chunks"]) == 1

