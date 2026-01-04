"""FastAPI application factory for BioRAG Bench."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from biorag import __version__
from biorag.api.dependencies import get_pipeline_manager
from biorag.api.routes import router
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


def create_app(
    config_path: str | Path | None = None,
    index_path: str | Path | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config_path: Path to configuration file
        index_path: Path to FAISS index directory
        cors_origins: List of allowed CORS origins

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="BioRAG Bench API",
        description="""
        Biomedical RAG (Retrieval-Augmented Generation) API for answering
        biomedical questions using PubMed literature.

        ## Features

        - **Question Answering**: Answer biomedical questions with citations
        - **Retrieval**: Retrieve relevant PubMed chunks for a query
        - **Structured Output**: JSON responses with answer, citations, and metadata

        ## Disclaimer

        ⚠️ **Medical Disclaimer**: This system is for research and educational purposes only.
        It should NOT be used for medical diagnosis, treatment decisions, or as a substitute
        for professional medical advice. Always consult qualified healthcare providers for
        medical questions.
        """,
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Add CORS middleware
    if cors_origins is None:
        cors_origins = ["*"]  # Allow all origins by default for development

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize pipeline manager
    manager = get_pipeline_manager()
    manager.initialize(config_path=config_path, index_path=index_path)

    # Include routes
    app.include_router(router, prefix="/api/v1")

    # Add root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> dict[str, Any]:
        """Root endpoint with API information."""
        return {
            "name": "BioRAG Bench API",
            "version": __version__,
            "description": "Biomedical RAG API for question answering",
            "docs_url": "/docs",
            "health_url": "/api/v1/health",
            "disclaimer": (
                "This system is for research and educational purposes only. "
                "It should NOT be used for medical diagnosis or treatment decisions."
            ),
        }

    # Startup event
    @app.on_event("startup")
    async def startup_event() -> None:
        """Initialize resources on startup."""
        logger.info(f"Starting BioRAG Bench API v{__version__}")
        # Pipeline is lazy-loaded on first request
        logger.info("Pipeline will be initialized on first request")

    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """Cleanup resources on shutdown."""
        logger.info("Shutting down BioRAG Bench API")

    return app


# Default app instance for uvicorn
app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    config_path: str | Path | None = None,
    index_path: str | Path | None = None,
    log_level: str = "info",
) -> None:
    """
    Run the FastAPI server using uvicorn.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
        config_path: Path to configuration file
        index_path: Path to FAISS index directory
        log_level: Logging level
    """
    import uvicorn

    # Initialize pipeline manager before running
    manager = get_pipeline_manager()
    manager.initialize(config_path=config_path, index_path=index_path)

    uvicorn.run(
        "biorag.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )

