"""Evaluation module for BioRAG Bench.

This module provides:
- Metrics: Recall@k, MRR, EM, F1, ROUGE-L, etc.
- Evaluators: BioASQ and PubMedQA specific evaluation
- Harness: End-to-end evaluation orchestration
"""

from biorag.eval.bioasq_eval import BioASQEvaluator, BioASQMetrics
from biorag.eval.harness import EvaluationHarness, EvalProgress
from biorag.eval.metrics import (
    accuracy,
    aggregate_scores,
    average_precision,
    confusion_matrix,
    exact_match,
    exact_match_any,
    macro_f1,
    mean_average_precision,
    mrr,
    normalize_answer,
    precision_at_k,
    recall_at_k,
    rouge_l,
    rouge_l_fmeasure,
    rouge_l_max,
    set_f1,
    token_f1,
    token_f1_max,
    tokenize,
)
from biorag.eval.pubmedqa_eval import PubMedQAEvaluator, PubMedQAMetrics

__all__ = [
    # Metrics
    "normalize_answer",
    "tokenize",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "average_precision",
    "mean_average_precision",
    "exact_match",
    "exact_match_any",
    "token_f1",
    "token_f1_max",
    "set_f1",
    "rouge_l",
    "rouge_l_fmeasure",
    "rouge_l_max",
    "accuracy",
    "macro_f1",
    "confusion_matrix",
    "aggregate_scores",
    # BioASQ
    "BioASQEvaluator",
    "BioASQMetrics",
    # PubMedQA
    "PubMedQAEvaluator",
    "PubMedQAMetrics",
    # Harness
    "EvaluationHarness",
    "EvalProgress",
]

