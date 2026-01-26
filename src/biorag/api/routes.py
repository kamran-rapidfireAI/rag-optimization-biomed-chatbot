"""FastAPI routes for BioRAG Bench API."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from biorag.api.dependencies import get_config, get_pipeline
from biorag.pipeline.rag import RAGPipeline
from biorag.schemas.config import BioRAGConfig
from biorag.schemas.evaluation import RetrievalResult
from biorag.schemas.generation import AnswerOutput, Citation

# Create API router
router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class AnswerRequest(BaseModel):
    """Request model for the /answer endpoint."""

    question: str = Field(..., description="The biomedical question to answer")
    question_type: str | None = Field(
        default=None,
        description="Type of question: yesno, factoid, list, or summary",
    )

    model_config = {"json_schema_extra": {"examples": [
        {
            "question": "What is the role of BRCA1 in breast cancer?",
            "question_type": "factoid",
        }
    ]}}


class ChunkResponse(BaseModel):
    """Response model for a retrieved chunk."""

    pmid: str = Field(..., description="PubMed ID")
    chunk_id: str = Field(..., description="Chunk identifier")
    text: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Retrieval score")
    rank: int = Field(..., description="Rank in results")
    rerank_score: float | None = Field(default=None, description="Reranking score")
    rerank_rank: int | None = Field(default=None, description="Rank after reranking")

    @classmethod
    def from_retrieval_result(cls, result: RetrievalResult) -> "ChunkResponse":
        """Create from RetrievalResult."""
        return cls(
            pmid=result.pmid,
            chunk_id=result.chunk_id,
            text=result.text,
            score=result.score,
            rank=result.rank,
            rerank_score=result.rerank_score,
            rerank_rank=result.rerank_rank,
        )


class LatencyBreakdown(BaseModel):
    """Latency breakdown for pipeline stages."""

    retrieve_ms: float = Field(..., description="Retrieval latency in ms")
    rerank_ms: float = Field(..., description="Reranking latency in ms")
    generate_ms: float = Field(..., description="Generation latency in ms")
    total_ms: float = Field(..., description="Total latency in ms")


class AnswerResponse(BaseModel):
    """Response model for the /answer endpoint."""

    answer: str = Field(..., description="The generated answer")
    answer_type: str = Field(..., description="Type of answer")
    label: str | None = Field(default=None, description="Yes/no/maybe for classification")
    confidence: float | None = Field(default=None, description="Model confidence")
    citations: list[Citation] = Field(default_factory=list, description="Citations")
    abstained: bool = Field(default=False, description="Whether model abstained")
    abstention_reason: str | None = Field(default=None, description="Reason for abstention")
    supported_by_evidence: bool = Field(default=True, description="Self-check result")

    # Retrieved evidence
    retrieved_chunks: list[ChunkResponse] = Field(
        default_factory=list, description="Retrieved and reranked chunks"
    )

    # Latency
    latency: LatencyBreakdown = Field(..., description="Latency breakdown")

    # Metadata
    model: str = Field(default="", description="LLM model used")
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")
    cache_hit: bool = Field(default=False, description="Whether response was cached")

    model_config = {"json_schema_extra": {"examples": [
        {
            "answer": "BRCA1 is a tumor suppressor gene...",
            "answer_type": "direct",
            "label": None,
            "confidence": 0.95,
            "citations": [{"pmid": "12345678", "chunk_id": "12345678_0", "quote": "..."}],
            "abstained": False,
            "abstention_reason": None,
            "supported_by_evidence": True,
            "retrieved_chunks": [],
            "latency": {
                "retrieve_ms": 50.0,
                "rerank_ms": 100.0,
                "generate_ms": 500.0,
                "total_ms": 650.0,
            },
            "model": "gpt-4o-mini",
            "input_tokens": 1500,
            "output_tokens": 200,
            "cache_hit": False,
        }
    ]}}


class RetrieveRequest(BaseModel):
    """Request model for the /retrieve endpoint."""

    question: str = Field(..., description="The query to retrieve documents for")
    k: int = Field(default=10, ge=1, le=100, description="Number of results")
    rerank: bool = Field(default=True, description="Whether to apply reranking")


class RetrieveResponse(BaseModel):
    """Response model for the /retrieve endpoint."""

    question: str = Field(..., description="Original query")
    chunks: list[ChunkResponse] = Field(..., description="Retrieved chunks")
    latency_ms: float = Field(..., description="Total retrieval latency in ms")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    pipeline_ready: bool = Field(..., description="Whether pipeline is ready")
    index_loaded: bool = Field(..., description="Whether FAISS index is loaded")
    config_summary: dict[str, Any] = Field(
        default_factory=dict, description="Configuration summary"
    )


class ConfigResponse(BaseModel):
    """Response model for configuration endpoint."""

    llm: dict[str, Any]
    embeddings: dict[str, Any]
    chunking: dict[str, Any]
    retrieval: dict[str, Any]
    rerank: dict[str, Any]


# ============================================================================
# Routes
# ============================================================================


@router.post("/answer", response_model=AnswerResponse, tags=["RAG"])
async def answer_question(
    request: AnswerRequest,
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
) -> AnswerResponse:
    """
    Answer a biomedical question using the RAG pipeline.

    This endpoint:
    1. Retrieves relevant chunks from the FAISS index
    2. Reranks chunks using a cross-encoder
    3. Generates an answer using an LLM with structured output

    Returns the answer with citations, retrieved evidence, and latency breakdown.
    """
    try:
        # Run the RAG pipeline
        result = pipeline.query(
            question=request.question,
            question_type=request.question_type,
        )

        # Build response
        answer = result.answer
        gen_response = result.generation_response

        return AnswerResponse(
            answer=answer.answer,
            answer_type=answer.answer_type,
            label=answer.label,
            confidence=answer.confidence,
            citations=answer.citations,
            abstained=answer.abstained,
            abstention_reason=answer.abstention_reason,
            supported_by_evidence=answer.supported_by_evidence,
            retrieved_chunks=[
                ChunkResponse.from_retrieval_result(chunk)
                for chunk in result.reranked_chunks
            ],
            latency=LatencyBreakdown(
                retrieve_ms=result.latency.retrieve_ms,
                rerank_ms=result.latency.rerank_ms,
                generate_ms=result.latency.generate_ms,
                total_ms=result.latency.total_ms,
            ),
            model=gen_response.model if gen_response else "",
            input_tokens=gen_response.input_tokens if gen_response else 0,
            output_tokens=gen_response.output_tokens if gen_response else 0,
            cache_hit=gen_response.cache_hit if gen_response else False,
        )

    except ValueError as e:
        # FAISS index not loaded
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline not ready: {e}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {e}",
        ) from e


@router.post("/retrieve", response_model=RetrieveResponse, tags=["RAG"])
async def retrieve_chunks(
    request: RetrieveRequest,
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
) -> RetrieveResponse:
    """
    Retrieve chunks for a query without generating an answer.

    Useful for debugging and understanding retrieval behavior.
    """
    try:
        import time

        start = time.perf_counter()

        if request.rerank:
            chunks = pipeline.retrieve_and_rerank(request.question)
        else:
            chunks = pipeline.retrieve_only(request.question)

        # Limit to requested k
        chunks = chunks[: request.k]
        latency_ms = (time.perf_counter() - start) * 1000

        return RetrieveResponse(
            question=request.question,
            chunks=[ChunkResponse.from_retrieval_result(c) for c in chunks],
            latency_ms=latency_ms,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline not ready: {e}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {e}",
        ) from e


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
) -> HealthResponse:
    """
    Check the health of the API and pipeline.

    Returns status information about the service and its components.
    """
    from biorag import __version__
    from biorag.api.dependencies import get_pipeline_manager

    manager = get_pipeline_manager()

    return HealthResponse(
        status="healthy",
        version=__version__,
        pipeline_ready=manager.is_initialized,
        index_loaded=manager.has_index,
        config_summary=pipeline.get_config_summary(),
    )


@router.get("/config", response_model=ConfigResponse, tags=["System"])
async def get_configuration(
    config: Annotated[BioRAGConfig, Depends(get_config)],
) -> ConfigResponse:
    """
    Get the current pipeline configuration.

    Returns the active configuration settings.
    """
    return ConfigResponse(
        llm={
            "provider": config.llm.provider,
            "model": config.llm.model,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
        },
        embeddings={
            "provider": config.embeddings.provider,
            "model": config.embeddings.model,
        },
        chunking={
            "type": config.chunking.type,
            "chunk_size": config.chunking.chunk_size,
            "chunk_overlap": config.chunking.chunk_overlap,
        },
        retrieval={
            "mode": config.retrieval.mode,
            "k": config.retrieval.k,
            "fetch_k": config.retrieval.fetch_k,
            "lambda_mult": config.retrieval.lambda_mult,
        },
        rerank={
            "enabled": config.rerank.enabled,
            "model": config.rerank.model,
            "top_n": config.rerank.top_n,
            "final_k": config.rerank.final_k,
        },
    )










