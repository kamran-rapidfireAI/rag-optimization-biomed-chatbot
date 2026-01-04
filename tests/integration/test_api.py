"""Integration tests for the FastAPI API."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from biorag.api.app import create_app
from biorag.api.dependencies import PipelineManager, get_pipeline_manager
from biorag.pipeline.rag import PipelineDebugInfo, PipelineLatency, RAGPipeline, RAGResult
from biorag.schemas.config import BioRAGConfig
from biorag.schemas.evaluation import RetrievalResult
from biorag.schemas.generation import AnswerOutput, GenerationResponse


@pytest.fixture
def mock_rag_result() -> RAGResult:
    """Create a mock RAG result for testing."""
    return RAGResult(
        answer=AnswerOutput(
            answer="BRCA1 is a tumor suppressor gene involved in DNA repair.",
            answer_type="direct",
            supported_by_evidence=True,
        ),
        retrieved_chunks=[
            RetrievalResult(
                pmid="12345678",
                chunk_id="12345678_0",
                text="BRCA1 plays a crucial role in maintaining genomic stability.",
                score=0.95,
                rank=1,
            ),
        ],
        reranked_chunks=[
            RetrievalResult(
                pmid="12345678",
                chunk_id="12345678_0",
                text="BRCA1 plays a crucial role in maintaining genomic stability.",
                score=0.95,
                rank=1,
                rerank_score=0.98,
                rerank_rank=1,
            ),
        ],
        latency=PipelineLatency(
            retrieve_ms=50.0,
            rerank_ms=100.0,
            generate_ms=200.0,
            total_ms=350.0,
        ),
        debug_info=PipelineDebugInfo(
            chunking={"type": "recursive"},
            retrieval={"mode": "mmr"},
            rerank={"enabled": True},
            generation={"model": "gpt-4o-mini"},
        ),
        generation_response=GenerationResponse(
            answer=AnswerOutput(
                answer="BRCA1 is a tumor suppressor gene involved in DNA repair.",
                answer_type="direct",
                supported_by_evidence=True,
            ),
            model="gpt-4o-mini",
            prompt_template="test",
            input_tokens=500,
            output_tokens=50,
            latency_ms=200.0,
            cache_hit=False,
        ),
    )


@pytest.fixture
def mock_pipeline(mock_rag_result: RAGResult) -> MagicMock:
    """Create a mock pipeline."""
    pipeline = MagicMock(spec=RAGPipeline)
    pipeline.query.return_value = mock_rag_result
    pipeline.retrieve_only.return_value = mock_rag_result.retrieved_chunks
    pipeline.retrieve_and_rerank.return_value = mock_rag_result.reranked_chunks
    pipeline.get_config_summary.return_value = {
        "llm": {"provider": "openai", "model": "gpt-4o-mini"},
        "embeddings": {"provider": "openai", "model": "text-embedding-3-large"},
        "chunking": {"type": "recursive", "chunk_size": 350},
        "retrieval": {"mode": "mmr", "k": 10},
        "rerank": {"enabled": True, "model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "final_k": 8},
    }
    return pipeline


@pytest.fixture
def mock_config() -> BioRAGConfig:
    """Create a mock config."""
    return BioRAGConfig()


@pytest.fixture
def mock_manager(
    mock_pipeline: MagicMock,
    mock_config: BioRAGConfig,
) -> MagicMock:
    """Create a mock pipeline manager."""
    manager = MagicMock(spec=PipelineManager)
    manager.get_pipeline.return_value = mock_pipeline
    manager.get_config.return_value = mock_config
    manager.is_initialized = True
    manager.has_index = True
    return manager


@pytest.fixture
def test_client(mock_manager: MagicMock) -> TestClient:
    """Create a test client with mocked dependencies."""
    app = create_app()
    
    # Override the pipeline manager dependency
    app.dependency_overrides[get_pipeline_manager] = lambda: mock_manager
    
    return TestClient(app)


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_info(self, test_client: TestClient) -> None:
        """Test root endpoint returns API info."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "BioRAG Bench API"
        assert "version" in data
        assert "disclaimer" in data


class TestHealthEndpoint:
    """Tests for the health endpoint."""

    def test_health_check(self, test_client: TestClient) -> None:
        """Test health endpoint returns status."""
        response = test_client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "pipeline_ready" in data
        assert "index_loaded" in data

    def test_health_includes_config_summary(
        self,
        test_client: TestClient,
    ) -> None:
        """Test health endpoint includes configuration summary."""
        response = test_client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "config_summary" in data


class TestConfigEndpoint:
    """Tests for the config endpoint."""

    def test_get_config(self, test_client: TestClient) -> None:
        """Test config endpoint returns configuration."""
        response = test_client.get("/api/v1/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "llm" in data
        assert "embeddings" in data
        assert "chunking" in data
        assert "retrieval" in data
        assert "rerank" in data


class TestAnswerEndpoint:
    """Tests for the /answer endpoint."""

    def test_answer_question(self, test_client: TestClient) -> None:
        """Test answering a biomedical question."""
        response = test_client.post(
            "/api/v1/answer",
            json={
                "question": "What is BRCA1?",
                "question_type": "factoid",
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "answer_type" in data
        assert "latency" in data
        assert "retrieved_chunks" in data

    def test_answer_without_question_type(
        self,
        test_client: TestClient,
    ) -> None:
        """Test answering without specifying question type."""
        response = test_client.post(
            "/api/v1/answer",
            json={"question": "What is cancer?"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_answer_includes_latency_breakdown(
        self,
        test_client: TestClient,
    ) -> None:
        """Test that answer includes latency breakdown."""
        response = test_client.post(
            "/api/v1/answer",
            json={"question": "What is BRCA1?"},
        )
        
        assert response.status_code == 200
        data = response.json()
        latency = data["latency"]
        assert "retrieve_ms" in latency
        assert "rerank_ms" in latency
        assert "generate_ms" in latency
        assert "total_ms" in latency

    def test_answer_includes_chunks(self, test_client: TestClient) -> None:
        """Test that answer includes retrieved chunks."""
        response = test_client.post(
            "/api/v1/answer",
            json={"question": "What is BRCA1?"},
        )
        
        assert response.status_code == 200
        data = response.json()
        chunks = data["retrieved_chunks"]
        assert len(chunks) > 0
        assert "pmid" in chunks[0]
        assert "text" in chunks[0]
        assert "score" in chunks[0]

    def test_answer_empty_question_validation(
        self,
        test_client: TestClient,
    ) -> None:
        """Test validation for empty question."""
        response = test_client.post(
            "/api/v1/answer",
            json={"question": ""},
        )
        
        # FastAPI/Pydantic should still accept empty string
        # The actual handling depends on implementation
        assert response.status_code in [200, 422]


class TestRetrieveEndpoint:
    """Tests for the /retrieve endpoint."""

    def test_retrieve_chunks(self, test_client: TestClient) -> None:
        """Test retrieving chunks for a query."""
        response = test_client.post(
            "/api/v1/retrieve",
            json={"question": "What is BRCA1?"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "question" in data
        assert "chunks" in data
        assert "latency_ms" in data

    def test_retrieve_with_custom_k(self, test_client: TestClient) -> None:
        """Test retrieving with custom k value."""
        response = test_client.post(
            "/api/v1/retrieve",
            json={
                "question": "What is BRCA1?",
                "k": 5,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["chunks"]) <= 5

    def test_retrieve_without_rerank(self, test_client: TestClient) -> None:
        """Test retrieving without reranking."""
        response = test_client.post(
            "/api/v1/retrieve",
            json={
                "question": "What is BRCA1?",
                "rerank": False,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data


class TestAPIErrorHandling:
    """Tests for API error handling."""

    def test_pipeline_not_ready(self, test_client: TestClient) -> None:
        """Test error when pipeline is not ready."""
        # Create a new client with a failing pipeline
        app = create_app()
        
        mock_manager = MagicMock(spec=PipelineManager)
        mock_pipeline = MagicMock(spec=RAGPipeline)
        mock_pipeline.query.side_effect = ValueError("FAISS store not initialized")
        mock_manager.get_pipeline.return_value = mock_pipeline
        mock_manager.get_config.return_value = BioRAGConfig()
        
        app.dependency_overrides[get_pipeline_manager] = lambda: mock_manager
        
        client = TestClient(app)
        response = client.post(
            "/api/v1/answer",
            json={"question": "What is BRCA1?"},
        )
        
        assert response.status_code == 503
        data = response.json()
        assert "Pipeline not ready" in data["detail"]

    def test_invalid_k_value(self, test_client: TestClient) -> None:
        """Test validation for invalid k value."""
        response = test_client.post(
            "/api/v1/retrieve",
            json={
                "question": "What is BRCA1?",
                "k": 0,
            },
        )
        
        assert response.status_code == 422  # Validation error


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, test_client: TestClient) -> None:
        """Test that CORS headers are present."""
        response = test_client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        
        # CORS preflight should return 200
        assert response.status_code == 200


class TestOpenAPI:
    """Tests for OpenAPI documentation."""

    def test_openapi_available(self, test_client: TestClient) -> None:
        """Test that OpenAPI schema is available."""
        response = test_client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_docs_available(self, test_client: TestClient) -> None:
        """Test that Swagger docs are available."""
        response = test_client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_redoc_available(self, test_client: TestClient) -> None:
        """Test that ReDoc is available."""
        response = test_client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

