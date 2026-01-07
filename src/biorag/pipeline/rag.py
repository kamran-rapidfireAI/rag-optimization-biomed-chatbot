"""RAG Pipeline for BioRAG Bench - orchestrates retrieve → rerank → generate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from biorag.embeddings.base import Embedder
from biorag.embeddings.local import LocalEmbedder
from biorag.embeddings.openai import OpenAIEmbedder
from biorag.generate.generator import Generator
from biorag.indexing.faiss_store import FAISSStore
from biorag.rerank.cross_encoder import CrossEncoderReranker
from biorag.retrieve.retriever import Retriever
from biorag.schemas.config import BioRAGConfig, load_config
from biorag.schemas.evaluation import RetrievalResult
from biorag.schemas.generation import AnswerOutput, GenerationRequest, GenerationResponse
from biorag.utils.caching import LLMCache
from biorag.utils.cost import CostTracker
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineLatency:
    """Latency breakdown for pipeline stages."""

    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "retrieve_ms": self.retrieve_ms,
            "rerank_ms": self.rerank_ms,
            "generate_ms": self.generate_ms,
            "total_ms": self.total_ms,
        }


@dataclass
class PipelineDebugInfo:
    """Debug information about pipeline configuration."""

    chunking: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    rerank: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunking": self.chunking,
            "retrieval": self.retrieval,
            "rerank": self.rerank,
            "generation": self.generation,
        }


@dataclass
class RAGResult:
    """Complete result from the RAG pipeline."""

    answer: AnswerOutput
    retrieved_chunks: list[RetrievalResult]
    reranked_chunks: list[RetrievalResult]
    latency: PipelineLatency
    debug_info: PipelineDebugInfo
    generation_response: GenerationResponse | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "answer": self.answer.model_dump(),
            "retrieved_chunks": [c.model_dump() for c in self.retrieved_chunks],
            "reranked_chunks": [c.model_dump() for c in self.reranked_chunks],
            "latency": self.latency.to_dict(),
            "debug_info": self.debug_info.to_dict(),
            "generation_response": (
                self.generation_response.model_dump()
                if self.generation_response
                else None
            ),
        }


class RAGPipeline:
    """
    End-to-end RAG pipeline orchestrating retrieve → rerank → generate.

    This pipeline:
    1. Retrieves relevant chunks from FAISS index
    2. Reranks chunks using a cross-encoder (optional)
    3. Generates an answer using an LLM with structured output
    """

    def __init__(
        self,
        config: BioRAGConfig | None = None,
        config_path: str | Path | None = None,
        faiss_store: FAISSStore | None = None,
        embedder: Embedder | None = None,
        retriever: Retriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        generator: Generator | None = None,
        cache: LLMCache | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """
        Initialize RAG pipeline.

        Args:
            config: BioRAG configuration
            config_path: Path to config file (used if config not provided)
            faiss_store: Pre-built FAISS store (optional)
            embedder: Pre-built embedder (optional)
            retriever: Pre-built retriever (optional)
            reranker: Pre-built reranker (optional)
            generator: Pre-built generator (optional)
            cache: LLM cache instance
            cost_tracker: Cost tracker instance
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            self.config = load_config()

        # Store injected components
        self._faiss_store = faiss_store
        self._embedder = embedder
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._cache = cache
        self._cost_tracker = cost_tracker

        # Build debug info from config
        self._debug_info = self._build_debug_info()

        logger.info("RAG Pipeline initialized")

    def _build_debug_info(self) -> PipelineDebugInfo:
        """Build debug info from configuration."""
        return PipelineDebugInfo(
            chunking={
                "type": self.config.chunking.type,
                "chunk_size": self.config.chunking.chunk_size,
                "chunk_overlap": self.config.chunking.chunk_overlap,
            },
            retrieval={
                "mode": self.config.retrieval.mode,
                "k": self.config.retrieval.k,
                "fetch_k": self.config.retrieval.fetch_k,
                "lambda_mult": self.config.retrieval.lambda_mult,
            },
            rerank={
                "enabled": self.config.rerank.enabled,
                "model": self.config.rerank.model,
                "top_n": self.config.rerank.top_n,
                "final_k": self.config.rerank.final_k,
            },
            generation={
                "model": self.config.llm.model,
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens,
            },
        )

    @property
    def embedder(self) -> Embedder:
        """Get or create embedder."""
        if self._embedder is None:
            self._embedder = self._create_embedder()
        return self._embedder

    @property
    def faiss_store(self) -> FAISSStore:
        """Get or create FAISS store."""
        if self._faiss_store is None:
            raise ValueError(
                "FAISS store not initialized. Either provide a faiss_store "
                "or call load_index() first."
            )
        return self._faiss_store

    @property
    def retriever(self) -> Retriever:
        """Get or create retriever."""
        if self._retriever is None:
            self._retriever = Retriever(
                store=self.faiss_store,
                mode=self.config.retrieval.mode,
                k=self.config.retrieval.k,
                fetch_k=self.config.retrieval.fetch_k,
                lambda_mult=self.config.retrieval.lambda_mult,
                score_threshold=self.config.retrieval.score_threshold,
            )
        return self._retriever

    @property
    def reranker(self) -> CrossEncoderReranker | None:
        """Get or create reranker (if enabled)."""
        if not self.config.rerank.enabled:
            return None
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                model=self.config.rerank.model,
                top_n=self.config.rerank.top_n,
                final_k=self.config.rerank.final_k,
                batch_size=self.config.rerank.batch_size,
            )
        return self._reranker

    @property
    def generator(self) -> Generator:
        """Get or create generator."""
        if self._generator is None:
            self._generator = Generator(
                config=self.config,
                cache=self._cache,
                cost_tracker=self._cost_tracker,
            )
        return self._generator

    def _create_embedder(self) -> Embedder:
        """Create embedder based on configuration."""
        if self.config.embeddings.provider == "openai":
            return OpenAIEmbedder(
                model=self.config.embeddings.model,
                batch_size=self.config.embeddings.batch_size,
            )
        elif self.config.embeddings.provider == "local":
            return LocalEmbedder()
        else:
            raise ValueError(f"Unknown embeddings provider: {self.config.embeddings.provider}")

    def load_index(self, index_path: str | Path) -> None:
        """
        Load FAISS index from disk.

        Args:
            index_path: Path to the index directory
        """
        index_path = Path(index_path)
        logger.info(f"Loading FAISS index from {index_path}")
        self._faiss_store = FAISSStore.load(index_path, self.embedder)
        # Reset retriever to use new store
        self._retriever = None

    def query(
        self,
        question: str,
        question_type: str | None = None,
        skip_generation: bool = False,
    ) -> RAGResult:
        """
        Execute the full RAG pipeline for a question.

        Args:
            question: The question to answer
            question_type: Type of question (yesno, factoid, list, summary)
            skip_generation: If True, only retrieve and rerank without generating

        Returns:
            RAGResult with answer, chunks, latency breakdown, and debug info
        """
        total_start = time.perf_counter()
        latency = PipelineLatency()

        # Stage 1: Retrieve
        retrieve_start = time.perf_counter()
        retrieved_chunks = self.retriever.retrieve(question)
        latency.retrieve_ms = (time.perf_counter() - retrieve_start) * 1000

        logger.log_pipeline_stage(
            "retrieve",
            {"query": question[:100], "num_results": len(retrieved_chunks)},
            latency.retrieve_ms,
        )

        # Stage 2: Rerank
        reranked_chunks = retrieved_chunks
        if self.reranker is not None and retrieved_chunks:
            rerank_start = time.perf_counter()
            reranked_chunks = self.reranker.rerank(question, retrieved_chunks)
            latency.rerank_ms = (time.perf_counter() - rerank_start) * 1000

            logger.log_pipeline_stage(
                "rerank",
                {
                    "input_count": len(retrieved_chunks),
                    "output_count": len(reranked_chunks),
                },
                latency.rerank_ms,
            )

        # Stage 3: Generate (optional)
        generation_response: GenerationResponse | None = None
        answer: AnswerOutput

        if skip_generation:
            answer = AnswerOutput(
                answer="",
                answer_type="unknown",
                abstained=True,
                abstention_reason="Generation skipped",
                supported_by_evidence=False,
            )
        else:
            generate_start = time.perf_counter()

            # Prepare evidence for generation
            evidence_chunks = [
                {
                    "pmid": chunk.pmid,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "score": chunk.rerank_score or chunk.score,
                    "rank": chunk.rerank_rank or chunk.rank,
                }
                for chunk in reranked_chunks
            ]

            request = GenerationRequest(
                question=question,
                evidence_chunks=evidence_chunks,
                question_type=question_type,
            )

            generation_response = self.generator.generate(request)
            answer = generation_response.answer
            latency.generate_ms = (time.perf_counter() - generate_start) * 1000

            logger.log_pipeline_stage(
                "generate",
                {
                    "model": generation_response.model,
                    "input_tokens": generation_response.input_tokens,
                    "output_tokens": generation_response.output_tokens,
                    "cache_hit": generation_response.cache_hit,
                    "abstained": answer.abstained,
                },
                latency.generate_ms,
            )

        latency.total_ms = (time.perf_counter() - total_start) * 1000

        return RAGResult(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            reranked_chunks=reranked_chunks,
            latency=latency,
            debug_info=self._debug_info,
            generation_response=generation_response,
        )

    def retrieve_only(self, question: str) -> list[RetrievalResult]:
        """
        Retrieve chunks without reranking or generation.

        Args:
            question: The question to retrieve for

        Returns:
            List of retrieved chunks
        """
        return self.retriever.retrieve(question)

    def retrieve_and_rerank(self, question: str) -> list[RetrievalResult]:
        """
        Retrieve and rerank chunks without generation.

        Args:
            question: The question to retrieve for

        Returns:
            List of reranked chunks
        """
        result = self.query(question, skip_generation=True)
        return result.reranked_chunks

    def get_config_summary(self) -> dict[str, Any]:
        """Get a summary of the pipeline configuration."""
        return {
            "llm": {
                "provider": self.config.llm.provider,
                "model": self.config.llm.model,
            },
            "embeddings": {
                "provider": self.config.embeddings.provider,
                "model": self.config.embeddings.model,
            },
            "chunking": {
                "type": self.config.chunking.type,
                "chunk_size": self.config.chunking.chunk_size,
            },
            "retrieval": {
                "mode": self.config.retrieval.mode,
                "k": self.config.retrieval.k,
            },
            "rerank": {
                "enabled": self.config.rerank.enabled,
                "model": self.config.rerank.model if self.config.rerank.enabled else None,
                "final_k": self.config.rerank.final_k if self.config.rerank.enabled else None,
            },
        }






