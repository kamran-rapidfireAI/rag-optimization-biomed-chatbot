"""Evaluation metrics for BioRAG Bench.

This module provides:
- Retrieval metrics: Recall@k, MRR, Precision@k, MAP
- Answer metrics: exact_match, token_f1, set_f1, ROUGE-L
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Sequence

from biorag.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Text Normalization Utilities
# =============================================================================


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.

    Applies:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Collapse whitespace

    Args:
        text: Raw answer text

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Collapse whitespace
    text = " ".join(text.split())

    return text.strip()


def tokenize(text: str) -> list[str]:
    """
    Tokenize text into words.

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


# =============================================================================
# Retrieval Metrics
# =============================================================================


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int | None = None,
) -> float:
    """
    Calculate Recall@k for retrieval.

    Recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: List of relevant document IDs (ground truth)
        k: Number of top results to consider (None = all)

    Returns:
        Recall@k score between 0 and 1
    """
    if not relevant:
        return 1.0  # No relevant docs, trivially satisfied

    retrieved_set = set(retrieved[:k] if k else retrieved)
    relevant_set = set(relevant)

    hits = len(retrieved_set & relevant_set)
    return hits / len(relevant_set)


def precision_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int | None = None,
) -> float:
    """
    Calculate Precision@k for retrieval.

    Precision@k = |retrieved[:k] ∩ relevant| / k

    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: List of relevant document IDs (ground truth)
        k: Number of top results to consider (None = len(retrieved))

    Returns:
        Precision@k score between 0 and 1
    """
    if k is None:
        k = len(retrieved)

    if k == 0:
        return 0.0

    retrieved_at_k = list(retrieved[:k])
    relevant_set = set(relevant)

    hits = sum(1 for doc in retrieved_at_k if doc in relevant_set)
    return hits / k


def mrr(
    retrieved: Sequence[str],
    relevant: Sequence[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for a single query.

    MRR = 1 / rank_of_first_relevant_doc

    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: List of relevant document IDs (ground truth)

    Returns:
        Reciprocal rank (0 if no relevant doc found)
    """
    if not relevant:
        return 1.0  # No relevant docs

    relevant_set = set(relevant)

    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def average_precision(
    retrieved: Sequence[str],
    relevant: Sequence[str],
) -> float:
    """
    Calculate Average Precision (AP) for a single query.

    AP = sum(P@k * rel(k)) / |relevant|

    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: List of relevant document IDs (ground truth)

    Returns:
        Average precision score
    """
    if not relevant:
        return 1.0

    relevant_set = set(relevant)
    score = 0.0
    num_hits = 0

    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            num_hits += 1
            precision = num_hits / rank
            score += precision

    return score / len(relevant_set)


def mean_average_precision(
    retrieved_lists: list[Sequence[str]],
    relevant_lists: list[Sequence[str]],
) -> float:
    """
    Calculate Mean Average Precision (MAP) across queries.

    Args:
        retrieved_lists: List of retrieved doc lists per query
        relevant_lists: List of relevant doc lists per query

    Returns:
        MAP score
    """
    if not retrieved_lists:
        return 0.0

    scores = [
        average_precision(retr, rel)
        for retr, rel in zip(retrieved_lists, relevant_lists)
    ]
    return sum(scores) / len(scores)


# =============================================================================
# Answer Metrics
# =============================================================================


def exact_match(prediction: str, reference: str) -> float:
    """
    Calculate exact match score.

    Args:
        prediction: Predicted answer
        reference: Gold reference answer

    Returns:
        1.0 if normalized answers match, 0.0 otherwise
    """
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def exact_match_any(prediction: str, references: Sequence[str]) -> float:
    """
    Calculate exact match against any reference.

    Args:
        prediction: Predicted answer
        references: List of acceptable reference answers

    Returns:
        1.0 if prediction matches any reference, 0.0 otherwise
    """
    if not references:
        return 0.0

    pred_normalized = normalize_answer(prediction)
    for ref in references:
        if pred_normalized == normalize_answer(ref):
            return 1.0
    return 0.0


def token_f1(prediction: str, reference: str) -> float:
    """
    Calculate token-level F1 score.

    Args:
        prediction: Predicted answer
        reference: Gold reference answer

    Returns:
        F1 score between 0 and 1
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    # Count common tokens
    common = sum((pred_counter & ref_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def token_f1_max(prediction: str, references: Sequence[str]) -> float:
    """
    Calculate maximum token F1 across references.

    Args:
        prediction: Predicted answer
        references: List of reference answers

    Returns:
        Maximum F1 score
    """
    if not references:
        return 0.0

    return max(token_f1(prediction, ref) for ref in references)


def set_f1(prediction_set: Sequence[str], reference_set: Sequence[str]) -> float:
    """
    Calculate set-level F1 for list-type questions.

    Args:
        prediction_set: Predicted set of answers
        reference_set: Gold reference set of answers

    Returns:
        F1 score between 0 and 1
    """
    if not prediction_set and not reference_set:
        return 1.0
    if not prediction_set or not reference_set:
        return 0.0

    # Normalize answers
    pred_normalized = {normalize_answer(a) for a in prediction_set if a}
    ref_normalized = {normalize_answer(a) for a in reference_set if a}

    if not pred_normalized and not ref_normalized:
        return 1.0
    if not pred_normalized or not ref_normalized:
        return 0.0

    # Count matches
    matches = len(pred_normalized & ref_normalized)

    if matches == 0:
        return 0.0

    precision = matches / len(pred_normalized)
    recall = matches / len(ref_normalized)

    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """
    Calculate ROUGE-L score using the rouge-score library.

    Args:
        prediction: Predicted answer
        reference: Gold reference answer

    Returns:
        Dictionary with precision, recall, and fmeasure
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)

        return {
            "precision": scores["rougeL"].precision,
            "recall": scores["rougeL"].recall,
            "fmeasure": scores["rougeL"].fmeasure,
        }
    except ImportError:
        logger.warning("rouge-score not installed, returning 0")
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
    except Exception as e:
        logger.warning(f"Error computing ROUGE-L: {e}")
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}


def rouge_l_fmeasure(prediction: str, reference: str) -> float:
    """
    Calculate ROUGE-L F-measure score.

    Args:
        prediction: Predicted answer
        reference: Gold reference answer

    Returns:
        ROUGE-L F-measure
    """
    return rouge_l(prediction, reference)["fmeasure"]


def rouge_l_max(prediction: str, references: Sequence[str]) -> float:
    """
    Calculate maximum ROUGE-L F-measure across references.

    Args:
        prediction: Predicted answer
        references: List of reference answers

    Returns:
        Maximum ROUGE-L F-measure
    """
    if not references:
        return 0.0

    return max(rouge_l_fmeasure(prediction, ref) for ref in references)


# =============================================================================
# Classification Metrics
# =============================================================================


def accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    """
    Calculate classification accuracy.

    Args:
        predictions: List of predicted labels
        references: List of gold labels

    Returns:
        Accuracy between 0 and 1
    """
    if not predictions:
        return 0.0

    correct = sum(
        1 for pred, ref in zip(predictions, references)
        if normalize_answer(pred) == normalize_answer(ref)
    )
    return correct / len(predictions)


def macro_f1(
    predictions: Sequence[str],
    references: Sequence[str],
    labels: Sequence[str] | None = None,
) -> float:
    """
    Calculate macro-averaged F1 score for classification.

    Args:
        predictions: List of predicted labels
        references: List of gold labels
        labels: List of unique labels (auto-detected if None)

    Returns:
        Macro F1 score
    """
    if not predictions:
        return 0.0

    # Normalize
    predictions = [normalize_answer(p) for p in predictions]
    references = [normalize_answer(r) for r in references]

    # Get unique labels
    if labels is None:
        labels = list(set(predictions) | set(references))
    else:
        labels = [normalize_answer(l) for l in labels]

    # Calculate per-class F1
    f1_scores = []
    for label in labels:
        tp = sum(1 for p, r in zip(predictions, references) if p == label and r == label)
        fp = sum(1 for p, r in zip(predictions, references) if p == label and r != label)
        fn = sum(1 for p, r in zip(predictions, references) if p != label and r == label)

        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def confusion_matrix(
    predictions: Sequence[str],
    references: Sequence[str],
    labels: Sequence[str] | None = None,
) -> dict[str, dict[str, int]]:
    """
    Calculate confusion matrix for classification.

    Args:
        predictions: List of predicted labels
        references: List of gold labels
        labels: List of unique labels (auto-detected if None)

    Returns:
        Nested dict: matrix[true_label][pred_label] = count
    """
    # Normalize
    predictions = [normalize_answer(p) for p in predictions]
    references = [normalize_answer(r) for r in references]

    # Get unique labels
    if labels is None:
        labels = sorted(set(predictions) | set(references))
    else:
        labels = [normalize_answer(l) for l in labels]

    # Initialize matrix
    matrix: dict[str, dict[str, int]] = {
        true_label: {pred_label: 0 for pred_label in labels}
        for true_label in labels
    }

    # Count
    for pred, ref in zip(predictions, references):
        if ref in matrix and pred in matrix[ref]:
            matrix[ref][pred] += 1

    return matrix


# =============================================================================
# Aggregate Metrics Helpers
# =============================================================================


def aggregate_scores(scores: Sequence[float]) -> dict[str, float]:
    """
    Aggregate a list of scores into summary statistics.

    Args:
        scores: List of individual scores

    Returns:
        Dict with mean, std, min, max, count
    """
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    import statistics

    scores_list = list(scores)
    n = len(scores_list)

    return {
        "mean": statistics.mean(scores_list),
        "std": statistics.stdev(scores_list) if n > 1 else 0.0,
        "min": min(scores_list),
        "max": max(scores_list),
        "count": n,
    }

