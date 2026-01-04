"""FastAPI dependencies for pipeline injection."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends

from biorag.pipeline.rag import RAGPipeline
from biorag.schemas.config import BioRAGConfig, load_config
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


class PipelineManager:
    """
    Manages the RAG pipeline singleton.
    
    This ensures we only load the FAISS index once and reuse
    the same pipeline instance across requests.
    """

    _instance: PipelineManager | None = None
    _pipeline: RAGPipeline | None = None
    _config: BioRAGConfig | None = None
    _index_path: Path | None = None

    def __new__(cls) -> PipelineManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(
        self,
        config_path: str | Path | None = None,
        index_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the pipeline with configuration.

        Args:
            config_path: Path to configuration file
            index_path: Path to FAISS index directory
        """
        self._config = load_config(config_path)
        
        if index_path is not None:
            self._index_path = Path(index_path)
        else:
            # Default index path
            self._index_path = self._config.paths.data_dir / "processed" / "index"
        
        logger.info(f"Pipeline manager initialized with config from {config_path}")

    def get_pipeline(self) -> RAGPipeline:
        """
        Get or create the RAG pipeline instance.

        Returns:
            RAGPipeline instance with loaded FAISS index
        """
        if self._pipeline is None:
            if self._config is None:
                self._config = load_config()
            
            self._pipeline = RAGPipeline(config=self._config)
            
            # Load FAISS index if path exists
            if self._index_path and self._index_path.exists():
                try:
                    self._pipeline.load_index(self._index_path)
                    logger.info(f"Loaded FAISS index from {self._index_path}")
                except Exception as e:
                    logger.warning(f"Could not load FAISS index: {e}")
            else:
                logger.warning(
                    f"FAISS index not found at {self._index_path}. "
                    "Retrieval will not work until index is loaded."
                )
        
        return self._pipeline

    def get_config(self) -> BioRAGConfig:
        """Get the current configuration."""
        if self._config is None:
            self._config = load_config()
        return self._config

    def reload_pipeline(self) -> RAGPipeline:
        """Force reload of the pipeline."""
        self._pipeline = None
        return self.get_pipeline()

    @property
    def is_initialized(self) -> bool:
        """Check if pipeline is initialized."""
        return self._pipeline is not None

    @property
    def has_index(self) -> bool:
        """Check if FAISS index is loaded."""
        if self._pipeline is None:
            return False
        try:
            _ = self._pipeline.faiss_store
            return True
        except ValueError:
            return False


@lru_cache()
def get_pipeline_manager() -> PipelineManager:
    """Get the singleton pipeline manager."""
    return PipelineManager()


def get_pipeline(
    manager: Annotated[PipelineManager, Depends(get_pipeline_manager)]
) -> RAGPipeline:
    """
    FastAPI dependency to get the RAG pipeline.

    Usage:
        @app.get("/endpoint")
        def endpoint(pipeline: Annotated[RAGPipeline, Depends(get_pipeline)]):
            ...
    """
    return manager.get_pipeline()


def get_config(
    manager: Annotated[PipelineManager, Depends(get_pipeline_manager)]
) -> BioRAGConfig:
    """
    FastAPI dependency to get the configuration.

    Usage:
        @app.get("/endpoint")
        def endpoint(config: Annotated[BioRAGConfig, Depends(get_config)]):
            ...
    """
    return manager.get_config()

