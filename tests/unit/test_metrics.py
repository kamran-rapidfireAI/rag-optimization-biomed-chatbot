"""Unit tests for evaluation metrics."""

from __future__ import annotations

import pytest

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
    set_f1,
    token_f1,
    token_f1_max,
    tokenize,
)


class TestNormalization:
    """Tests for text normalization utilities."""

    def test_normalize_answer_basic(self) -> None:
        """Test basic normalization."""
        assert normalize_answer("Hello World") == "hello world"

    def test_normalize_answer_punctuation(self) -> None:
        """Test punctuation removal."""
        assert normalize_answer("Hello, World!") == "hello world"

    def test_normalize_answer_articles(self) -> None:
        """Test article removal."""
        assert normalize_answer("The quick brown fox") == "quick brown fox"
        assert normalize_answer("A cat and an apple") == "cat and apple"

    def test_normalize_answer_whitespace(self) -> None:
        """Test whitespace collapsing."""
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_normalize_answer_empty(self) -> None:
        """Test empty string."""
        assert normalize_answer("") == ""
        assert normalize_answer("   ") == ""

    def test_tokenize_basic(self) -> None:
        """Test basic tokenization."""
        assert tokenize("hello world") == ["hello", "world"]

    def test_tokenize_with_normalization(self) -> None:
        """Test tokenization applies normalization."""
        assert tokenize("The QUICK, brown fox!") == ["quick", "brown", "fox"]

    def test_tokenize_empty(self) -> None:
        """Test empty tokenization."""
        assert tokenize("") == []


class TestRetrievalMetrics:
    """Tests for retrieval metrics."""

    def test_recall_at_k_basic(self) -> None:
        """Test basic recall@k."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc6"]

        assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(1 / 3)
        assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)
        assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 3)

    def test_recall_at_k_all_relevant(self) -> None:
        """Test recall when all relevant docs retrieved."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2"]

        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_recall_at_k_none_relevant(self) -> None:
        """Test recall when no relevant docs retrieved."""
        retrieved = ["doc4", "doc5", "doc6"]
        relevant = ["doc1", "doc2", "doc3"]

        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_recall_at_k_empty_relevant(self) -> None:
        """Test recall with empty relevant set."""
        assert recall_at_k(["doc1", "doc2"], [], k=2) == 1.0

    def test_precision_at_k_basic(self) -> None:
        """Test basic precision@k."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3"]

        assert precision_at_k(retrieved, relevant, k=1) == 1.0  # doc1 is relevant
        assert precision_at_k(retrieved, relevant, k=2) == 0.5  # doc1 relevant, doc2 not
        assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)
        assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 5)

    def test_precision_at_k_empty(self) -> None:
        """Test precision with empty retrieval."""
        assert precision_at_k([], ["doc1"], k=5) == 0.0

    def test_mrr_basic(self) -> None:
        """Test basic MRR."""
        # First relevant at rank 1
        assert mrr(["doc1", "doc2", "doc3"], ["doc1"]) == 1.0
        # First relevant at rank 2
        assert mrr(["doc2", "doc1", "doc3"], ["doc1"]) == 0.5
        # First relevant at rank 3
        assert mrr(["doc2", "doc3", "doc1"], ["doc1"]) == pytest.approx(1 / 3)

    def test_mrr_multiple_relevant(self) -> None:
        """Test MRR with multiple relevant docs."""
        # Returns reciprocal rank of first relevant
        assert mrr(["doc2", "doc1", "doc3"], ["doc1", "doc3"]) == 0.5

    def test_mrr_no_relevant(self) -> None:
        """Test MRR when no relevant doc found."""
        assert mrr(["doc1", "doc2"], ["doc3"]) == 0.0

    def test_mrr_empty_relevant(self) -> None:
        """Test MRR with empty relevant set."""
        assert mrr(["doc1", "doc2"], []) == 1.0

    def test_average_precision_basic(self) -> None:
        """Test average precision."""
        # Retrieved: [rel, non, rel, non, non]
        retrieved = ["doc1", "doc5", "doc2", "doc6", "doc7"]
        relevant = ["doc1", "doc2", "doc3"]

        # P@1 = 1/1 (doc1 relevant)
        # P@3 = 2/3 (doc1, doc2 relevant)
        # AP = (1/1 + 2/3) / 3 = (1 + 0.667) / 3 = 0.556
        ap = average_precision(retrieved, relevant)
        assert ap == pytest.approx((1 + 2 / 3) / 3)

    def test_average_precision_perfect(self) -> None:
        """Test AP with perfect retrieval."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3"]

        # P@1 = 1, P@2 = 1, P@3 = 1
        # AP = 3/3 = 1
        assert average_precision(retrieved, relevant) == 1.0

    def test_mean_average_precision(self) -> None:
        """Test MAP across queries."""
        retrieved_lists = [
            ["doc1", "doc2", "doc3"],  # Perfect for first query
            ["doc4", "doc1", "doc2"],  # doc1 at rank 2 for second query
        ]
        relevant_lists = [
            ["doc1"],
            ["doc1"],
        ]

        # Query 1: AP = 1.0
        # Query 2: AP = 0.5
        # MAP = (1.0 + 0.5) / 2 = 0.75
        assert mean_average_precision(retrieved_lists, relevant_lists) == pytest.approx(0.75)


class TestAnswerMetrics:
    """Tests for answer metrics."""

    def test_exact_match_basic(self) -> None:
        """Test basic exact match."""
        assert exact_match("Paris", "Paris") == 1.0
        assert exact_match("Paris", "London") == 0.0

    def test_exact_match_normalization(self) -> None:
        """Test exact match with normalization."""
        assert exact_match("The Paris", "paris") == 1.0
        assert exact_match("HELLO, World!", "hello world") == 1.0

    def test_exact_match_empty(self) -> None:
        """Test exact match with empty strings."""
        assert exact_match("", "") == 1.0
        assert exact_match("hello", "") == 0.0

    def test_exact_match_any_basic(self) -> None:
        """Test exact match against multiple references."""
        assert exact_match_any("Paris", ["Paris", "France"]) == 1.0
        assert exact_match_any("France", ["Paris", "France"]) == 1.0
        assert exact_match_any("London", ["Paris", "France"]) == 0.0

    def test_exact_match_any_empty(self) -> None:
        """Test exact match any with empty references."""
        assert exact_match_any("test", []) == 0.0

    def test_token_f1_identical(self) -> None:
        """Test token F1 with identical strings."""
        assert token_f1("hello world", "hello world") == 1.0

    def test_token_f1_partial(self) -> None:
        """Test token F1 with partial overlap."""
        # Pred: [hello, world] Ref: [hello, there]
        # Common: 1, Precision: 1/2, Recall: 1/2
        # F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        assert token_f1("hello world", "hello there") == pytest.approx(0.5)

    def test_token_f1_no_overlap(self) -> None:
        """Test token F1 with no overlap."""
        assert token_f1("hello world", "goodbye moon") == 0.0

    def test_token_f1_empty(self) -> None:
        """Test token F1 with empty strings."""
        assert token_f1("", "") == 1.0
        assert token_f1("hello", "") == 0.0
        assert token_f1("", "hello") == 0.0

    def test_token_f1_max_basic(self) -> None:
        """Test token F1 max across references."""
        assert token_f1_max("Paris", ["Paris", "London"]) == 1.0
        assert token_f1_max("Paris city", ["Paris", "London"]) == pytest.approx(2 / 3)

    def test_set_f1_identical(self) -> None:
        """Test set F1 with identical sets."""
        assert set_f1(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_set_f1_partial(self) -> None:
        """Test set F1 with partial overlap."""
        # Pred: {a, b}, Ref: {a, c}
        # Matches: 1, Precision: 1/2, Recall: 1/2
        # F1 = 0.5
        assert set_f1(["a", "b"], ["a", "c"]) == pytest.approx(0.5)

    def test_set_f1_no_overlap(self) -> None:
        """Test set F1 with no overlap."""
        assert set_f1(["a", "b"], ["c", "d"]) == 0.0

    def test_set_f1_empty(self) -> None:
        """Test set F1 with empty sets."""
        assert set_f1([], []) == 1.0
        assert set_f1(["a"], []) == 0.0
        assert set_f1([], ["a"]) == 0.0

    def test_set_f1_normalization(self) -> None:
        """Test set F1 normalizes items."""
        assert set_f1(["The Paris", "LONDON"], ["paris", "london"]) == 1.0


class TestRougeL:
    """Tests for ROUGE-L metric."""

    def test_rouge_l_identical(self) -> None:
        """Test ROUGE-L with identical strings."""
        scores = rouge_l("hello world", "hello world")
        assert scores["fmeasure"] == pytest.approx(1.0, rel=0.01)

    def test_rouge_l_partial(self) -> None:
        """Test ROUGE-L with partial overlap."""
        scores = rouge_l("hello world", "hello there world")
        assert 0 < scores["fmeasure"] < 1

    def test_rouge_l_no_overlap(self) -> None:
        """Test ROUGE-L with no overlap."""
        scores = rouge_l("abc", "xyz")
        assert scores["fmeasure"] == pytest.approx(0.0, abs=0.1)

    def test_rouge_l_fmeasure(self) -> None:
        """Test ROUGE-L F-measure helper."""
        score = rouge_l_fmeasure("hello world", "hello world")
        assert score == pytest.approx(1.0, rel=0.01)


class TestClassificationMetrics:
    """Tests for classification metrics."""

    def test_accuracy_basic(self) -> None:
        """Test basic accuracy."""
        predictions = ["yes", "no", "yes", "yes"]
        references = ["yes", "no", "no", "yes"]

        assert accuracy(predictions, references) == pytest.approx(3 / 4)

    def test_accuracy_perfect(self) -> None:
        """Test perfect accuracy."""
        predictions = ["yes", "no", "maybe"]
        references = ["yes", "no", "maybe"]

        assert accuracy(predictions, references) == 1.0

    def test_accuracy_empty(self) -> None:
        """Test accuracy with empty lists."""
        assert accuracy([], []) == 0.0

    def test_macro_f1_basic(self) -> None:
        """Test basic macro F1."""
        predictions = ["yes", "yes", "no", "no", "maybe"]
        references = ["yes", "no", "no", "no", "maybe"]

        # Calculate per-class:
        # yes: TP=1, FP=1, FN=0 -> P=0.5, R=1.0, F1=0.67
        # no: TP=2, FP=0, FN=1 -> P=1.0, R=0.67, F1=0.8
        # maybe: TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
        f1 = macro_f1(predictions, references, labels=["yes", "no", "maybe"])
        assert 0.7 < f1 < 0.9  # Approximate

    def test_macro_f1_binary(self) -> None:
        """Test macro F1 for binary classification."""
        predictions = ["yes", "yes", "no", "no"]
        references = ["yes", "no", "yes", "no"]

        f1 = macro_f1(predictions, references, labels=["yes", "no"])
        assert f1 == pytest.approx(0.5)

    def test_confusion_matrix_basic(self) -> None:
        """Test basic confusion matrix."""
        predictions = ["yes", "yes", "no", "no"]
        references = ["yes", "no", "yes", "no"]

        matrix = confusion_matrix(predictions, references, labels=["yes", "no"])

        # Expected:
        #        pred_yes  pred_no
        # true_yes    1       1
        # true_no     1       1
        assert matrix["yes"]["yes"] == 1
        assert matrix["yes"]["no"] == 1
        assert matrix["no"]["yes"] == 1
        assert matrix["no"]["no"] == 1


class TestAggregation:
    """Tests for score aggregation."""

    def test_aggregate_scores_basic(self) -> None:
        """Test basic score aggregation."""
        scores = [0.5, 0.7, 0.9, 0.8, 0.6]
        result = aggregate_scores(scores)

        assert result["mean"] == pytest.approx(0.7)
        assert result["min"] == 0.5
        assert result["max"] == 0.9
        assert result["count"] == 5
        assert result["std"] > 0

    def test_aggregate_scores_single(self) -> None:
        """Test aggregation with single score."""
        result = aggregate_scores([0.8])

        assert result["mean"] == 0.8
        assert result["std"] == 0.0
        assert result["count"] == 1

    def test_aggregate_scores_empty(self) -> None:
        """Test aggregation with empty list."""
        result = aggregate_scores([])

        assert result["mean"] == 0.0
        assert result["count"] == 0

