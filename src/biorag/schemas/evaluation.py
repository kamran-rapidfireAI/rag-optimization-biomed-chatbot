"""Evaluation schemas for BioRAG Bench."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class BioASQQuestion(BaseModel):
    """Schema for a BioASQ question."""

    question_id: str = Field(..., description="Unique question identifier")
    question_text: str = Field(..., description="The question text")
    question_type: Literal["yesno", "factoid", "list", "summary"] = Field(
        ..., description="Type of question"
    )

    # Gold documents and snippets
    gold_pmids: list[str] = Field(
        default_factory=list, description="Gold standard PubMed IDs"
    )
    gold_snippets: list[str] = Field(
        default_factory=list, description="Gold standard text snippets"
    )

    # Gold answers
    exact_answer: str | list[str] | None = Field(
        default=None, description="Exact answer(s) for factoid/list/yesno"
    )
    ideal_answer: str | None = Field(
        default=None, description="Ideal answer for summary questions"
    )

    # Metadata
    year: int | None = Field(default=None, description="BioASQ challenge year")
    batch: str | None = Field(default=None, description="BioASQ batch identifier")

    model_config = {"extra": "ignore"}


class PubMedQAQuestion(BaseModel):
    """Schema for a PubMedQA question."""

    question_id: str = Field(..., description="Unique question identifier (PMID)")
    question_text: str = Field(..., description="The question text")
    
    # Context (from the paper)
    context: list[str] = Field(
        default_factory=list, description="Context sentences from the paper"
    )
    long_answer: str = Field(default="", description="Long answer from the paper")

    # Gold label
    label: Literal["yes", "no", "maybe"] = Field(..., description="Gold standard label")

    # Source
    pmid: str = Field(..., description="Source PubMed ID")
    split: Literal["train", "dev", "test"] = Field(
        default="test", description="Dataset split"
    )

    model_config = {"extra": "ignore"}


class RetrievalResult(BaseModel):
    """Result from retrieval stage."""

    pmid: str = Field(..., description="Retrieved document PMID")
    chunk_id: str = Field(..., description="Retrieved chunk ID")
    text: str = Field(..., description="Retrieved text content")
    score: float = Field(..., description="Retrieval score")
    rank: int = Field(..., description="Rank in results (1-indexed)")

    # After reranking
    rerank_score: float | None = Field(default=None, description="Reranker score")
    rerank_rank: int | None = Field(default=None, description="Rank after reranking")

    model_config = {"extra": "ignore"}


class EvalPrediction(BaseModel):
    """A single prediction for evaluation."""

    question_id: str = Field(..., description="Question identifier")
    
    # Retrieval info
    retrieved_pmids: list[str] = Field(
        default_factory=list, description="Retrieved PMIDs in order"
    )
    retrieved_chunks: list[RetrievalResult] = Field(
        default_factory=list, description="Retrieved chunks with scores"
    )

    # Answer info
    predicted_answer: str = Field(default="", description="Generated answer")
    predicted_label: str | None = Field(
        default=None, description="Predicted label for classification"
    )
    abstained: bool = Field(default=False, description="Whether the model abstained")
    abstention_reason: str | None = Field(
        default=None, description="Reason for abstention"
    )

    # Latency
    retrieval_latency_ms: float = Field(default=0.0, description="Retrieval latency")
    rerank_latency_ms: float = Field(default=0.0, description="Reranking latency")
    generation_latency_ms: float = Field(default=0.0, description="Generation latency")
    total_latency_ms: float = Field(default=0.0, description="Total latency")

    # Token usage
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")

    # Raw output for debugging
    raw_output: dict[str, Any] | None = Field(
        default=None, description="Raw LLM output for debugging"
    )

    model_config = {"extra": "ignore"}


class MetricResult(BaseModel):
    """Result for a single metric."""

    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    count: int = Field(default=0, description="Number of samples evaluated")
    std: float | None = Field(default=None, description="Standard deviation if applicable")

    model_config = {"extra": "ignore"}


class RunMetrics(BaseModel):
    """Aggregated metrics for an evaluation run."""

    # Run info
    run_id: str = Field(..., description="Unique run identifier")
    run_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the run was executed"
    )

    # Dataset info
    dataset: str = Field(..., description="Dataset evaluated on")
    num_questions: int = Field(..., description="Number of questions evaluated")
    num_abstained: int = Field(default=0, description="Number of abstentions")

    # Retrieval metrics
    retrieval_metrics: dict[str, MetricResult] = Field(
        default_factory=dict, description="Retrieval metrics (Recall@k, MRR, etc.)"
    )

    # Answer metrics
    answer_metrics: dict[str, MetricResult] = Field(
        default_factory=dict, description="Answer metrics (EM, F1, ROUGE, etc.)"
    )

    # Latency stats
    avg_retrieval_latency_ms: float = Field(default=0.0)
    avg_rerank_latency_ms: float = Field(default=0.0)
    avg_generation_latency_ms: float = Field(default=0.0)
    avg_total_latency_ms: float = Field(default=0.0)

    # Cost stats
    total_input_tokens: int = Field(default=0)
    total_output_tokens: int = Field(default=0)
    estimated_cost_usd: float = Field(default=0.0)
    cache_hit_rate: float = Field(default=0.0)

    model_config = {"extra": "ignore"}


class EvalResult(BaseModel):
    """Complete evaluation result including predictions and metrics."""

    # Run info
    run_id: str = Field(..., description="Unique run identifier")
    
    # Config used
    config: dict[str, Any] = Field(
        default_factory=dict, description="Configuration used for the run"
    )

    # Reproducibility info
    git_commit: str | None = Field(default=None, description="Git commit SHA")
    dataset_version: str | None = Field(default=None, description="Dataset version")
    model_versions: dict[str, str] = Field(
        default_factory=dict, description="Model versions used"
    )
    random_seed: int = Field(default=42, description="Random seed used")

    # Results
    predictions: list[EvalPrediction] = Field(
        default_factory=list, description="Individual predictions"
    )
    metrics: RunMetrics | None = Field(
        default=None, description="Aggregated metrics"
    )

    model_config = {"extra": "ignore"}

