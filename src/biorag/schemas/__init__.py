"""Pydantic schemas for BioRAG Bench."""

from biorag.schemas.config import BioRAGConfig, load_config
from biorag.schemas.corpus import Chunk, CorpusDocument, CorpusManifest
from biorag.schemas.evaluation import (
    BioASQQuestion,
    EvalPrediction,
    EvalResult,
    MetricResult,
    PubMedQAQuestion,
    RetrievalResult,
    RunMetrics,
)
from biorag.schemas.generation import (
    AnswerOutput,
    Citation,
    GenerationRequest,
    GenerationResponse,
)

__all__ = [
    # Config
    "BioRAGConfig",
    "load_config",
    # Corpus
    "Chunk",
    "CorpusDocument",
    "CorpusManifest",
    # Evaluation
    "BioASQQuestion",
    "EvalPrediction",
    "EvalResult",
    "MetricResult",
    "PubMedQAQuestion",
    "RetrievalResult",
    "RunMetrics",
    # Generation
    "AnswerOutput",
    "Citation",
    "GenerationRequest",
    "GenerationResponse",
]
