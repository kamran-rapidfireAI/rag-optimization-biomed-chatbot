"""Abstention logic for BioRAG Bench generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from biorag.schemas.evaluation import RetrievalResult
from biorag.schemas.generation import AnswerOutput
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


class AbstentionReason(str, Enum):
    """Reasons for abstention."""
    
    # Evidence-based
    NO_EVIDENCE = "no_evidence"
    LOW_EVIDENCE_SCORE = "low_evidence_score"
    INSUFFICIENT_CHUNKS = "insufficient_chunks"
    
    # Model self-check
    MODEL_UNCERTAIN = "model_uncertain"
    UNSUPPORTED_BY_EVIDENCE = "unsupported_by_evidence"
    
    # Combined
    SCORE_AND_SELF_CHECK = "score_and_self_check"
    
    # Other
    CONFLICTING_EVIDENCE = "conflicting_evidence"


@dataclass
class AbstentionConfig:
    """Configuration for abstention logic."""
    
    # Evidence score thresholds
    min_evidence_score: float = 0.3
    min_evidence_chunks: int = 1
    
    # Model self-check
    enable_self_check: bool = True
    
    # Whether to abstain when evidence conflicts
    abstain_on_conflict: bool = False


@dataclass
class AbstentionDecision:
    """Result of abstention check."""
    
    should_abstain: bool
    reason: AbstentionReason | None = None
    details: str | None = None
    evidence_scores: list[float] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "should_abstain": self.should_abstain,
            "reason": self.reason.value if self.reason else None,
            "details": self.details,
            "evidence_scores": self.evidence_scores,
        }


class AbstentionChecker:
    """
    Checks whether the model should abstain from answering.
    
    Two-stage checking:
    1. Pre-generation: Check evidence quality before calling LLM
    2. Post-generation: Check model's self-assessment in output
    """

    def __init__(self, config: AbstentionConfig | None = None) -> None:
        """
        Initialize abstention checker.

        Args:
            config: Abstention configuration
        """
        self.config = config or AbstentionConfig()

    def check_evidence(
        self,
        chunks: list[RetrievalResult] | list[dict[str, Any]],
    ) -> AbstentionDecision:
        """
        Check evidence quality before generation.
        
        This is a pre-generation check to avoid wasting LLM calls
        on questions with poor evidence.

        Args:
            chunks: Retrieved evidence chunks

        Returns:
            AbstentionDecision indicating whether to abstain
        """
        # No evidence at all
        if not chunks:
            return AbstentionDecision(
                should_abstain=True,
                reason=AbstentionReason.NO_EVIDENCE,
                details="No evidence chunks retrieved",
                evidence_scores=[],
            )
        
        # Extract scores
        scores = self._extract_scores(chunks)
        
        # Check minimum chunk count
        if len(chunks) < self.config.min_evidence_chunks:
            return AbstentionDecision(
                should_abstain=True,
                reason=AbstentionReason.INSUFFICIENT_CHUNKS,
                details=f"Only {len(chunks)} chunks, minimum is {self.config.min_evidence_chunks}",
                evidence_scores=scores,
            )
        
        # Check minimum score threshold
        # Use the best score (after reranking if available)
        if scores and max(scores) < self.config.min_evidence_score:
            return AbstentionDecision(
                should_abstain=True,
                reason=AbstentionReason.LOW_EVIDENCE_SCORE,
                details=f"Best score {max(scores):.3f} < threshold {self.config.min_evidence_score}",
                evidence_scores=scores,
            )
        
        # Evidence passes pre-generation checks
        return AbstentionDecision(
            should_abstain=False,
            evidence_scores=scores,
        )

    def check_model_output(
        self,
        output: AnswerOutput,
        evidence_decision: AbstentionDecision | None = None,
    ) -> AbstentionDecision:
        """
        Check model's self-assessment in output.
        
        This is a post-generation check that respects the model's
        own judgment about evidence support.

        Args:
            output: Generated answer output
            evidence_decision: Previous evidence-based decision

        Returns:
            Final AbstentionDecision
        """
        # If model already abstained
        if output.abstained:
            return AbstentionDecision(
                should_abstain=True,
                reason=AbstentionReason.MODEL_UNCERTAIN,
                details=output.abstention_reason or "Model chose to abstain",
                evidence_scores=evidence_decision.evidence_scores if evidence_decision else None,
            )
        
        # Check model's self-check flag (if enabled)
        if self.config.enable_self_check and not output.supported_by_evidence:
            # Combine with evidence decision if both indicate problems
            if evidence_decision and evidence_decision.evidence_scores:
                scores = evidence_decision.evidence_scores
                if scores and max(scores) < self.config.min_evidence_score:
                    return AbstentionDecision(
                        should_abstain=True,
                        reason=AbstentionReason.SCORE_AND_SELF_CHECK,
                        details="Low evidence scores and model reports unsupported",
                        evidence_scores=scores,
                    )
            
            return AbstentionDecision(
                should_abstain=True,
                reason=AbstentionReason.UNSUPPORTED_BY_EVIDENCE,
                details="Model indicates answer not fully supported by evidence",
                evidence_scores=evidence_decision.evidence_scores if evidence_decision else None,
            )
        
        # Answer is valid
        return AbstentionDecision(
            should_abstain=False,
            evidence_scores=evidence_decision.evidence_scores if evidence_decision else None,
        )

    def should_abstain(
        self,
        chunks: list[RetrievalResult] | list[dict[str, Any]],
        output: AnswerOutput | None = None,
    ) -> AbstentionDecision:
        """
        Full abstention check combining evidence and model assessment.

        Args:
            chunks: Retrieved evidence chunks
            output: Optional generated output for post-generation check

        Returns:
            AbstentionDecision with final verdict
        """
        # Pre-generation check
        evidence_decision = self.check_evidence(chunks)
        
        if evidence_decision.should_abstain:
            return evidence_decision
        
        # Post-generation check (if output provided)
        if output is not None:
            return self.check_model_output(output, evidence_decision)
        
        return evidence_decision

    def _extract_scores(
        self,
        chunks: list[RetrievalResult] | list[dict[str, Any]],
    ) -> list[float]:
        """Extract scores from chunks, preferring rerank scores."""
        scores = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                # Prefer rerank_score over score
                score = chunk.get("rerank_score") or chunk.get("score")
            else:
                score = chunk.rerank_score if chunk.rerank_score is not None else chunk.score
            
            if score is not None:
                scores.append(float(score))
        
        return scores


def apply_abstention(
    output: AnswerOutput,
    decision: AbstentionDecision,
) -> AnswerOutput:
    """
    Apply abstention decision to an output.
    
    Modifies the output to reflect abstention if decided.

    Args:
        output: The generated output
        decision: The abstention decision

    Returns:
        Modified output with abstention applied
    """
    if not decision.should_abstain:
        return output
    
    # Create abstained version
    return AnswerOutput(
        answer="I cannot answer this question based on the available evidence.",
        answer_type="abstained",
        label=None,
        confidence=0.0,
        citations=[],
        abstained=True,
        abstention_reason=decision.details or (decision.reason.value if decision.reason else "Unknown"),
        supported_by_evidence=False,
        answer_list=None,
    )





