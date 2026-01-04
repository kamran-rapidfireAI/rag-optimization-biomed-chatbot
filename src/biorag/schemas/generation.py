"""Generation schemas for BioRAG Bench."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A citation to a source document."""

    pmid: str = Field(..., description="PubMed ID of the cited document")
    chunk_id: str | None = Field(default=None, description="Specific chunk ID if available")
    quote: str | None = Field(
        default=None, description="Relevant quote from the source"
    )
    relevance_score: float | None = Field(
        default=None, description="How relevant this citation is"
    )

    model_config = {"extra": "ignore"}


class AnswerOutput(BaseModel):
    """Structured output from the generation stage."""

    # Core answer
    answer: str = Field(..., description="The generated answer text")
    answer_type: Literal["direct", "abstained", "partial", "unknown"] = Field(
        default="direct", description="Type of answer"
    )

    # For yes/no/maybe questions
    label: Literal["yes", "no", "maybe"] | None = Field(
        default=None, description="Predicted label for classification questions"
    )
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Confidence in the answer"
    )

    # Citations
    citations: list[Citation] = Field(
        default_factory=list, description="Citations supporting the answer"
    )

    # Abstention
    abstained: bool = Field(default=False, description="Whether the model abstained")
    abstention_reason: str | None = Field(
        default=None, description="Reason for abstention"
    )
    
    # Self-check
    supported_by_evidence: bool = Field(
        default=True, description="Model's self-assessment of evidence support"
    )

    # For list-type questions
    answer_list: list[str] | None = Field(
        default=None, description="List of answers for list-type questions"
    )

    model_config = {"extra": "ignore"}

    def has_valid_citations(self, required_pmids: set[str] | None = None) -> bool:
        """Check if all citations reference valid PMIDs."""
        if not self.citations:
            return False
        
        for citation in self.citations:
            if not citation.pmid:
                return False
            if required_pmids and citation.pmid not in required_pmids:
                return False
        
        return True


class GenerationRequest(BaseModel):
    """Request for answer generation."""

    question: str = Field(..., description="The question to answer")
    evidence_chunks: list[dict] = Field(
        default_factory=list, description="Retrieved evidence chunks"
    )
    question_type: str | None = Field(
        default=None, description="Type of question (yesno, factoid, list, summary)"
    )

    # Options
    require_citations: bool = Field(
        default=True, description="Whether to require citations"
    )
    allow_abstention: bool = Field(
        default=True, description="Whether to allow abstention"
    )

    model_config = {"extra": "ignore"}


class GenerationResponse(BaseModel):
    """Response from generation including metadata."""

    answer: AnswerOutput = Field(..., description="The generated answer")
    
    # Metadata
    model: str = Field(default="", description="Model used for generation")
    prompt_template: str = Field(default="", description="Prompt template used")
    
    # Token usage
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")
    
    # Latency
    latency_ms: float = Field(default=0.0, description="Generation latency in ms")
    
    # Caching
    cache_hit: bool = Field(default=False, description="Whether this was a cache hit")

    model_config = {"extra": "ignore"}

