"""BioASQ evaluator for BioRAG Bench.

Implements evaluation for all BioASQ question types:
- yesno: accuracy
- factoid: exact match, token-F1
- list: set-F1
- summary: ROUGE-L (and optional BERTScore)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from biorag.eval.metrics import (
    accuracy,
    aggregate_scores,
    exact_match,
    exact_match_any,
    mrr,
    recall_at_k,
    rouge_l_fmeasure,
    set_f1,
    token_f1,
    token_f1_max,
)
from biorag.schemas.evaluation import BioASQQuestion, EvalPrediction, MetricResult
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BioASQMetrics:
    """Aggregated metrics for BioASQ evaluation."""

    # Overall metrics
    num_questions: int = 0
    num_abstained: int = 0

    # Per-type counts
    yesno_count: int = 0
    factoid_count: int = 0
    list_count: int = 0
    summary_count: int = 0

    # Retrieval metrics
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0

    # Answer metrics (aggregated across types)
    overall_score: float = 0.0  # Weighted average of type-specific scores

    # Per-type metrics
    yesno_accuracy: float = 0.0
    factoid_em: float = 0.0
    factoid_f1: float = 0.0
    list_f1: float = 0.0
    summary_rouge_l: float = 0.0

    # Detailed per-type results
    yesno_results: list[dict[str, Any]] = field(default_factory=list)
    factoid_results: list[dict[str, Any]] = field(default_factory=list)
    list_results: list[dict[str, Any]] = field(default_factory=list)
    summary_results: list[dict[str, Any]] = field(default_factory=list)

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
            "yesno_accuracy": MetricResult(
                name="yesno_accuracy", value=self.yesno_accuracy, count=self.yesno_count
            ),
            "factoid_em": MetricResult(
                name="factoid_em", value=self.factoid_em, count=self.factoid_count
            ),
            "factoid_f1": MetricResult(
                name="factoid_f1", value=self.factoid_f1, count=self.factoid_count
            ),
            "list_f1": MetricResult(
                name="list_f1", value=self.list_f1, count=self.list_count
            ),
            "summary_rouge_l": MetricResult(
                name="summary_rouge_l", value=self.summary_rouge_l, count=self.summary_count
            ),
            "overall_score": MetricResult(
                name="overall_score", value=self.overall_score, count=self.num_questions
            ),
        }


class BioASQEvaluator:
    """
    Evaluator for BioASQ benchmark questions.

    Supports all four question types with appropriate metrics:
    - yesno: accuracy
    - factoid: exact match, token-F1
    - list: set-F1
    - summary: ROUGE-L
    """

    def __init__(
        self,
        use_bert_score: bool = False,
        strict_matching: bool = False,
    ) -> None:
        """
        Initialize BioASQ evaluator.

        Args:
            use_bert_score: Whether to use BERTScore for summary evaluation
            strict_matching: Whether to use strict matching for factoid/list
        """
        self.use_bert_score = use_bert_score
        self.strict_matching = strict_matching

        # Try to import BERTScore if requested
        if use_bert_score:
            try:
                import bert_score  # noqa: F401
                self._has_bert_score = True
            except ImportError:
                logger.warning("BERTScore requested but not installed")
                self._has_bert_score = False
        else:
            self._has_bert_score = False

    def evaluate(
        self,
        questions: Sequence[BioASQQuestion],
        predictions: Sequence[EvalPrediction],
    ) -> BioASQMetrics:
        """
        Evaluate predictions against gold questions.

        Args:
            questions: List of gold BioASQ questions
            predictions: List of predictions (matched by question_id)

        Returns:
            BioASQMetrics with aggregated results
        """
        # Build prediction lookup
        pred_map = {p.question_id: p for p in predictions}

        # Initialize metrics
        metrics = BioASQMetrics(num_questions=len(questions))

        # Per-type score accumulators
        yesno_scores: list[float] = []
        factoid_em_scores: list[float] = []
        factoid_f1_scores: list[float] = []
        list_scores: list[float] = []
        summary_scores: list[float] = []

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

            # Evaluate retrieval
            if question.gold_pmids:
                recall_1_scores.append(
                    recall_at_k(pred.retrieved_pmids, question.gold_pmids, k=1)
                )
                recall_5_scores.append(
                    recall_at_k(pred.retrieved_pmids, question.gold_pmids, k=5)
                )
                recall_10_scores.append(
                    recall_at_k(pred.retrieved_pmids, question.gold_pmids, k=10)
                )
                mrr_scores.append(mrr(pred.retrieved_pmids, question.gold_pmids))

            # Evaluate answer by type
            if question.question_type == "yesno":
                metrics.yesno_count += 1
                result = self._evaluate_yesno(question, pred)
                yesno_scores.append(result["score"])
                metrics.yesno_results.append(result)

            elif question.question_type == "factoid":
                metrics.factoid_count += 1
                result = self._evaluate_factoid(question, pred)
                factoid_em_scores.append(result["em"])
                factoid_f1_scores.append(result["f1"])
                metrics.factoid_results.append(result)

            elif question.question_type == "list":
                metrics.list_count += 1
                result = self._evaluate_list(question, pred)
                list_scores.append(result["score"])
                metrics.list_results.append(result)

            elif question.question_type == "summary":
                metrics.summary_count += 1
                result = self._evaluate_summary(question, pred)
                summary_scores.append(result["rouge_l"])
                metrics.summary_results.append(result)

        # Aggregate retrieval metrics
        if recall_1_scores:
            metrics.recall_at_1 = sum(recall_1_scores) / len(recall_1_scores)
        if recall_5_scores:
            metrics.recall_at_5 = sum(recall_5_scores) / len(recall_5_scores)
        if recall_10_scores:
            metrics.recall_at_10 = sum(recall_10_scores) / len(recall_10_scores)
        if mrr_scores:
            metrics.mrr = sum(mrr_scores) / len(mrr_scores)

        # Aggregate answer metrics
        if yesno_scores:
            metrics.yesno_accuracy = sum(yesno_scores) / len(yesno_scores)
        if factoid_em_scores:
            metrics.factoid_em = sum(factoid_em_scores) / len(factoid_em_scores)
        if factoid_f1_scores:
            metrics.factoid_f1 = sum(factoid_f1_scores) / len(factoid_f1_scores)
        if list_scores:
            metrics.list_f1 = sum(list_scores) / len(list_scores)
        if summary_scores:
            metrics.summary_rouge_l = sum(summary_scores) / len(summary_scores)

        # Calculate overall weighted score
        total_weight = 0.0
        weighted_sum = 0.0

        if yesno_scores:
            weighted_sum += metrics.yesno_accuracy * len(yesno_scores)
            total_weight += len(yesno_scores)
        if factoid_f1_scores:
            weighted_sum += metrics.factoid_f1 * len(factoid_f1_scores)
            total_weight += len(factoid_f1_scores)
        if list_scores:
            weighted_sum += metrics.list_f1 * len(list_scores)
            total_weight += len(list_scores)
        if summary_scores:
            weighted_sum += metrics.summary_rouge_l * len(summary_scores)
            total_weight += len(summary_scores)

        if total_weight > 0:
            metrics.overall_score = weighted_sum / total_weight

        return metrics

    def _evaluate_yesno(
        self,
        question: BioASQQuestion,
        prediction: EvalPrediction,
    ) -> dict[str, Any]:
        """Evaluate a yes/no question."""
        gold = str(question.exact_answer).lower() if question.exact_answer else ""
        pred = (prediction.predicted_label or prediction.predicted_answer).lower()

        # Normalize to yes/no
        if "yes" in pred:
            pred = "yes"
        elif "no" in pred:
            pred = "no"
        else:
            pred = pred.strip()

        score = 1.0 if pred == gold else 0.0

        return {
            "question_id": question.question_id,
            "gold": gold,
            "predicted": pred,
            "score": score,
        }

    def _evaluate_factoid(
        self,
        question: BioASQQuestion,
        prediction: EvalPrediction,
    ) -> dict[str, Any]:
        """Evaluate a factoid question."""
        # Gold answers can be a single string or list of synonyms
        gold_answers: list[str] = []
        if isinstance(question.exact_answer, str):
            gold_answers = [question.exact_answer]
        elif isinstance(question.exact_answer, list):
            gold_answers = [str(a) for a in question.exact_answer]

        predicted = prediction.predicted_answer

        # Calculate EM (match any gold answer)
        em = exact_match_any(predicted, gold_answers)

        # Calculate F1 (max against any gold answer)
        f1 = token_f1_max(predicted, gold_answers)

        return {
            "question_id": question.question_id,
            "gold": gold_answers,
            "predicted": predicted,
            "em": em,
            "f1": f1,
        }

    def _evaluate_list(
        self,
        question: BioASQQuestion,
        prediction: EvalPrediction,
    ) -> dict[str, Any]:
        """Evaluate a list question."""
        # Gold answers
        gold_list: list[str] = []
        if isinstance(question.exact_answer, list):
            gold_list = [str(a) for a in question.exact_answer]
        elif isinstance(question.exact_answer, str) and question.exact_answer:
            gold_list = [question.exact_answer]

        # Predicted list from answer_list field or parse from answer
        predicted_list: list[str] = []
        # Try to get from raw_output first
        if prediction.raw_output and "answer_list" in prediction.raw_output:
            raw_list = prediction.raw_output["answer_list"]
            if isinstance(raw_list, list):
                predicted_list = [str(a) for a in raw_list]

        # Fall back to parsing predicted_answer
        if not predicted_list and prediction.predicted_answer:
            # Try to split by common delimiters
            answer = prediction.predicted_answer
            if "," in answer:
                predicted_list = [a.strip() for a in answer.split(",")]
            elif ";" in answer:
                predicted_list = [a.strip() for a in answer.split(";")]
            elif "\n" in answer:
                predicted_list = [a.strip() for a in answer.split("\n")]
            else:
                predicted_list = [answer.strip()]

        # Filter empty strings
        predicted_list = [a for a in predicted_list if a]

        score = set_f1(predicted_list, gold_list)

        return {
            "question_id": question.question_id,
            "gold": gold_list,
            "predicted": predicted_list,
            "score": score,
        }

    def _evaluate_summary(
        self,
        question: BioASQQuestion,
        prediction: EvalPrediction,
    ) -> dict[str, Any]:
        """Evaluate a summary question."""
        gold = question.ideal_answer or ""
        predicted = prediction.predicted_answer

        rouge = rouge_l_fmeasure(predicted, gold)

        result = {
            "question_id": question.question_id,
            "gold": gold[:200] + "..." if len(gold) > 200 else gold,
            "predicted": predicted[:200] + "..." if len(predicted) > 200 else predicted,
            "rouge_l": rouge,
        }

        # Optionally add BERTScore
        if self._has_bert_score:
            try:
                import bert_score

                P, R, F = bert_score.score(
                    [predicted],
                    [gold],
                    lang="en",
                    verbose=False,
                )
                result["bert_score_f1"] = F.item()
            except Exception as e:
                logger.warning(f"Error computing BERTScore: {e}")
                result["bert_score_f1"] = 0.0

        return result

    def evaluate_single(
        self,
        question: BioASQQuestion,
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
        if question.question_type == "yesno":
            return self._evaluate_yesno(question, prediction)
        elif question.question_type == "factoid":
            return self._evaluate_factoid(question, prediction)
        elif question.question_type == "list":
            return self._evaluate_list(question, prediction)
        elif question.question_type == "summary":
            return self._evaluate_summary(question, prediction)
        else:
            raise ValueError(f"Unknown question type: {question.question_type}")

