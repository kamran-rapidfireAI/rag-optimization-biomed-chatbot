"""Corpus and chunk schemas for BioRAG Bench."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class CorpusDocument(BaseModel):
    """Represents a document in the corpus (e.g., PubMed abstract)."""

    pmid: str = Field(..., description="PubMed ID")
    title: str = Field(default="", description="Article title")
    abstract: str = Field(..., description="Abstract text")
    authors: list[str] = Field(default_factory=list, description="List of authors")
    journal: str = Field(default="", description="Journal name")
    year: int | None = Field(default=None, description="Publication year")
    mesh_terms: list[str] = Field(default_factory=list, description="MeSH terms")
    keywords: list[str] = Field(default_factory=list, description="Author keywords")

    # Provenance
    source: Literal["pubmed", "bioasq", "pubmedqa"] = Field(
        default="pubmed", description="Source of the document"
    )
    is_gold: bool = Field(
        default=False, description="Whether this is a gold document for evaluation"
    )

    def full_text(self) -> str:
        """Return combined title and abstract."""
        if self.title:
            return f"{self.title}\n\n{self.abstract}"
        return self.abstract

    model_config = {"extra": "ignore"}


class Chunk(BaseModel):
    """Represents a chunk of text from a corpus document."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    pmid: str = Field(..., description="Source PubMed ID")
    text: str = Field(..., description="Chunk text content")
    
    # Position info
    start_char: int = Field(default=0, description="Start character offset in source")
    end_char: int = Field(default=0, description="End character offset in source")
    chunk_index: int = Field(default=0, description="Index of this chunk in the document")
    total_chunks: int = Field(default=1, description="Total chunks in the document")

    # Metadata
    section: str | None = Field(
        default=None, description="Section tag (title, abstract, etc.)"
    )
    token_count: int | None = Field(default=None, description="Token count if available")

    # Embedding (populated during indexing)
    embedding: list[float] | None = Field(
        default=None, exclude=True, description="Vector embedding"
    )

    model_config = {"extra": "ignore"}

    @property
    def doc_id(self) -> str:
        """Return the document identifier (PMID)."""
        return self.pmid


class CorpusManifest(BaseModel):
    """Manifest file for corpus provenance and reproducibility."""

    # Build info
    build_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When corpus was built"
    )
    build_version: str = Field(default="1.0.0", description="Corpus build version")

    # Source info
    source_method: Literal["huggingface", "pubmed_api", "local"] = Field(
        default="huggingface", description="How documents were obtained"
    )
    dataset_name: str = Field(default="", description="HuggingFace dataset name")
    dataset_version: str = Field(default="", description="Dataset version")
    dataset_revision: str = Field(default="", description="Git commit/revision hash")
    dataset_splits: list[str] = Field(
        default_factory=list, description="Dataset splits used"
    )

    # Sampling config
    sampling_seed: int = Field(default=42, description="Random seed for sampling")
    gold_pmid_count: int = Field(default=0, description="Number of gold PMIDs")
    distractor_pmid_count: int = Field(default=0, description="Number of distractor PMIDs")
    total_records: int = Field(default=0, description="Total records materialized")

    # Filtering
    min_abstract_length: int = Field(default=100, description="Minimum abstract length")
    max_abstract_length: int | None = Field(default=None, description="Maximum abstract length")

    # Output files
    output_files: dict[str, str] = Field(
        default_factory=dict, description="Output file paths and checksums"
    )

    model_config = {"extra": "ignore"}

