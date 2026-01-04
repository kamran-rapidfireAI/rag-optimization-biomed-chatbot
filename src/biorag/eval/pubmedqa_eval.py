"""PubMedQA evaluator for BioRAG Bench.

Implements evaluation for PubMedQA benchmark:
- Label prediction accuracy (yes/no/maybe)
- Macro-F1 across labels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from biorag.eval.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    mrr,
    normalize_answer,
    recall_at_k,
)
from biorag.schemas.evaluation import EvalPrediction, MetricResult, PubMedQAQuestion
from biorag.utils.logging import get_logger

logger = get_logger(__name__)

LABELS: list[Literal["yes", "no", "maybe"]] = ["yes", "no", "maybe"]


@dataclass
class PubMedQAMetrics:
    """Aggregated metrics for PubMedQA evaluation."""

    # Overall counts
    num_questions: int = 0
    num_abstained: int = 0

    # Retrieval metrics
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0

    # Classification metrics
    accuracy: float = 0.0
    macro_f1: float = 0.0

    # Per-label metrics
    yes_precision: float = 0.0
    yes_recall: float = 0.0
    yes_f1: float = 0.0
    no_precision: float = 0.0
    no_recall: float = 0.0
    no_f1: float = 0.0
    maybe_precision: float = 0.0
    maybe_recall: float = 0.0
    maybe_f1: float = 0.0

    # Confusion matrix
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    # Detailed results
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_metric_results(self) -> dict[str, MetricResult]:
        """Convert to MetricResult format."""
        return {
            "recall@1": MetricResult(
                name="recall@1", value=self.recall_at_1, count=self.num_questions
            ),
            "recall@5": MetricResult(
                name="recall@5", value=self.recall_at_5, count=self.num_questions
            ),
            "recall@10": MetricResult(
                name="recall@10", value=self.recall_at_10, count=self.num_questions
            ),
            "mrr": MetricResult(name="mrr", value=self.mrr, count=self.num_questions),
            "accuracy": MetricResult(
                name="accuracy", value=self.accuracy, count=self.num_questions
            ),
            "macro_f1": MetricResult(
                name="macro_f1", value=self.macro_f1, count=self.num_questions
            ),
            "yes_f1": MetricResult(
                name="yes_f1", value=self.yes_f1, count=self.num_questions
            ),
            "no_f1": MetricResult(
                name="no_f1", value=self.no_f1, count=self.num_questions
            ),
            "maybe_f1": MetricResult(
                name="maybe_f1", value=self.maybe_f1, count=self.num_questions
            ),
        }


class PubMedQAEvaluator:
    """
    Evaluator for PubMedQA benchmark.

    Evaluates yes/no/maybe label prediction with:
    - Accuracy
    - Macro-F1
    - Per-label precision/recall/F1
    """

    def __init__(
        self,
        include_maybe: bool = True,
    ) -> None:
        """
        Initialize PubMedQA evaluator.

        Args:
            include_maybe: Whether to include "maybe" as a valid label
        """
        self.include_maybe = include_maybe
        self.labels = LABELS if include_maybe else ["yes", "no"]

    def evaluate(
        self,
        questions: Sequence[PubMedQAQuestion],
        predictions: Sequence[EvalPrediction],
    ) -> PubMedQAMetrics:
        """
        Evaluate predictions against gold questions.

        Args:
            questions: List of gold PubMedQA questions
            predictions: List of predictions (matched by question_id)

        Returns:
            PubMedQAMetrics with aggregated results
        """
        # Build prediction lookup
        pred_map = {p.question_id: p for p in predictions}

        # Initialize metrics
        metrics = PubMedQAMetrics(num_questions=len(questions))

        # Collect predictions and golds for aggregate metrics
        gold_labels: list[str] = []
        pred_labels: list[str] = []

        # Retrieval metrics accumulators
        recall_1_scores: list[float] = []
        recall_5_scores: list[float] = []
        recall_10_scores: list[float] = []
        mrr_scores: list[float] = []

        for question in questions:
            pred = pred_map.get(question.question_id)

            if pred is None:
                logger.warning(f"No prediction for question {question.question_id}")
                continue

            if pred.abstained:
                metrics.num_abstained += 1

            # Evaluate retrieval (against the source PMID)
            gold_pmids = [question.pmid]
            if gold_pmids[0]:
                recall_1_scores.append(
                    recall_at_k(pred.retrieved_pmids, gold_pmids, k=1)
                )
                recall_5_scores.append(
                    recall_at_k(pred.retrieved_pmids, gold_pmids, k=5)
                )
                recall_10_scores.append(
                    recall_at_k(pred.retrieved_pmids, gold_pmids, k=10)
                )
                mrr_scores.append(mrr(pred.retrieved_pmids, gold_pmids))

            # Get predicted label
            predicted_label = self._extract_label(pred)

            # Collect for aggregate metrics
            gold_labels.append(question.label)
            pred_labels.append(predicted_label)

            # Store individual result
            result = {
                "question_id": question.question_id,
                "gold": question.label,
                "predicted": predicted_label,
                "correct": predicted_label == question.label,
            }
            metrics.results.append(result)

        # Aggregate retrieval metrics
        if recall_1_scores:
            metrics.recall_at_1 = sum(recall_1_scores) / len(recall_1_scores)
        if recall_5_scores:
            metrics.recall_at_5 = sum(recall_5_scores) / len(recall_5_scores)
        if recall_10_scores:
            metrics.recall_at_10 = sum(recall_10_scores) / len(recall_10_scores)
        if mrr_scores:
            metrics.mrr = sum(mrr_scores) / len(mrr_scores)

        # Classification metrics
        if gold_labels and pred_labels:
            metrics.accuracy = accuracy(pred_labels, gold_labels)
            metrics.macro_f1 = macro_f1(pred_labels, gold_labels, labels=self.labels)
            metrics.confusion = confusion_matrix(pred_labels, gold_labels, labels=self.labels)

            # Calculate per-label metrics
            for label in self.labels:
                tp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == label and g == label)
                fp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == label and g != label)
                fn = sum(1 for p, g in zip(pred_labels, gold_labels) if p != label and g == label)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                if label == "yes":
                    metrics.yes_precision = precision
                    metrics.yes_recall = recall
                    metrics.yes_f1 = f1
                elif label == "no":
                    metrics.no_precision = precision
                    metrics.no_recall = recall
                    metrics.no_f1 = f1
                elif label == "maybe":
                    metrics.maybe_precision = precision
                    metrics.maybe_recall = recall
                    metrics.maybe_f1 = f1

        return metrics

    def _extract_label(self, prediction: EvalPrediction) -> str:
        """
        Extract predicted label from prediction.

        Args:
            prediction: Model prediction

        Returns:
            Predicted label (yes, no, or maybe)
        """
        # First try predicted_label field
        if prediction.predicted_label:
            label = normalize_answer(prediction.predicted_label)
            if label in self.labels:
                return label

        # Fall back to parsing predicted_answer
        answer = normalize_answer(prediction.predicted_answer)

        # Check for explicit labels
        if "yes" in answer and "no" not in answer:
            return "yes"
        elif "no" in answer and "yes" not in answer:
            return "no"
        elif "maybe" in answer or "uncertain" in answer or "unclear" in answer:
            return "maybe" if self.include_maybe else "no"

        # Try to match first word
        first_word = answer.split()[0] if answer.split() else ""
        if first_word in self.labels:
            return first_word

        # Default to maybe (or no if maybe not included)
        return "maybe" if self.include_maybe else "no"

    def evaluate_single(
        self,
        question: PubMedQAQuestion,
        prediction: EvalPrediction,
    ) -> dict[str, Any]:
        """
        Evaluate a single question-prediction pair.

        Args:
            question: Gold question
            prediction: Model prediction

        Returns:
            Dictionary with evaluation results
        """
        predicted_label = self._extract_label(prediction)

        return {
            "question_id": question.question_id,
            "gold": question.label,
            "predicted": predicted_label,
            "correct": predicted_label == question.label,
        }

