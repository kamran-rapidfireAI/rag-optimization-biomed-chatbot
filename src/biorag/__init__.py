"""
BioRAG Bench - Biomedical RAG Optimization Pipeline

A comprehensive pipeline for building, evaluating, and optimizing
Retrieval-Augmented Generation (RAG) systems for biomedical question answering.
"""

__version__ = "0.1.0"
__author__ = "BioRAG Team"

from biorag.schemas.config import BioRAGConfig, load_config

__all__ = [
    "__version__",
    "BioRAGConfig",
    "load_config",
]

