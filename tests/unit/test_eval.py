"""Unit tests for BioASQ and PubMedQA evaluators."""

from __future__ import annotations

import pytest

from biorag.eval.bioasq_eval import BioASQEvaluator, BioASQMetrics
from biorag.eval.pubmedqa_eval import PubMedQAEvaluator, PubMedQAMetrics
from biorag.schemas.evaluation import (
    BioASQQuestion,
    EvalPrediction,
    PubMedQAQuestion,
)


class TestBioASQEvaluator:
    """Tests for BioASQ evaluator."""

    @pytest.fixture
    def evaluator(self) -> BioASQEvaluator:
        """Create evaluator instance."""
        return BioASQEvaluator(use_bert_score=False)

    def test_evaluate_yesno_correct(self, evaluator: BioASQEvaluator) -> None:
        """Test yes/no evaluation with correct answer."""
        question = BioASQQuestion(
            question_id="q1",
            question_text="Is aspirin used for pain relief?",
            question_type="yesno",
            exact_answer="yes",
            gold_pmids=["12345"],
        )
        prediction = EvalPrediction(
            question_id="q1",
            predicted_answer="Yes, aspirin is commonly used for pain relief.",
            predicted_label="yes",
            retrieved_pmids=["12345", "67890"],
        )

        result = evaluator._evaluate_yesno(question, prediction)
        assert result["score"] == 1.0
        assert result["predicted"] == "yes"
        assert result["gold"] == "yes"

    def test_evaluate_yesno_incorrect(self, evaluator: BioASQEvaluator) -> None:
        """Test yes/no evaluation with incorrect answer."""
        question = BioASQQuestion(
            question_id="q1",
            question_text="Is aspirin toxic in all doses?",
            question_type="yesno",
            exact_answer="no",
        )
        prediction = EvalPrediction(
            question_id="q1",
            predicted_answer="Yes, it is toxic.",
            predicted_label="yes",
        )

        result = evaluator._evaluate_yesno(question, prediction)
        assert result["score"] == 0.0

    def test_evaluate_factoid_exact_match(self, evaluator: BioASQEvaluator) -> None:
        """Test factoid evaluation with exact match."""
        question = BioASQQuestion(
            question_id="q1",
            question_text="What is the capital of France?",
            question_type="factoid",
            exact_answer="Paris",
        )
        prediction = EvalPrediction(
            question_id="q1",
            predicted_answer="Paris",
        )

        result = evaluator._evaluate_factoid(question, prediction)
        assert result["em"] == 1.0
        assert result["f1"] == 1.0

    def test_evaluate_factoid_partial_match(self, evaluator: BioASQEvaluator) -> None:
        """Test factoid evaluation with partial match."""
        question = BioASQQuestion(
            question_id="q1",
            question_text="What is the capital of France?",
            question_type="factoid",
            exact_answer="Paris",
        )
        prediction = EvalPrediction(
            question_id="q1",
            predicted_answer="The capital is Paris, France.",
        )

        result = evaluator._evaluate_factoid(question, prediction)
        assert result["em"] == 0.0  # Not exact
        assert result["f1"] > 0.0  # But has token overlap

    def test_evaluate_factoid_multiple_answers(self, evaluator: BioASQEvaluator) -> None:
        """Test factoid with multiple acceptable answers."""
        question = BioASQQuestion(
            question_id="q1",
            question_text="What is aspirin also called?",
            question_type="factoid",
            exact_answer=["acetylsalicylic acid", "ASA"],
        )
        prediction = EvalPrediction(
            question_id="q1",
            predicted_answer="ASA",
        )

        result = evaluator._evaluate_factoid(question, prediction)
        assert result["em"] == 1.0

    def test_evaluate_list(self, evaluator: BioASQEvaluator) -> None:
        """Test list evaluation."""
        question = BioASQQuestion(
            question_id="q1",
            question_text="Name three pain relievers.",
            question_type="list",
            exact_answer=["aspirin", "ibuprofen", "acetaminophen"],
        )
        prediction = EvalPrediction(
            question_id="q1",
            predicted_answer="aspirin, ibuprofen, naproxen",
        )

        result = evaluator._evaluate_list(question, prediction)
        # 2 out of 3 predicted are correct, 2 out of 3 gold are found
        # set_f1 = 2 * (2/3) * (2/3) / (2/3 + 2/3) = 2/3
        assert result["score"] == pytest.approx(2 / 3)

    def test_evaluate_list_perfect(self, evaluator: BioASQEvaluator) -> None:
        """Test list evaluation with perfect match."""
        question = BioASQQuestion(
            question_id="q1",
            question_text="Name symptoms of flu.",
            question_type="list",
            exact_answer=["fever", "cough"],
        )
        prediction = EvalPrediction(
            question_id="q1",
            predicted_answer="fever, cough",
        )

        result = evaluator._evaluate_list(question, prediction)
        assert result["score"] == 1.0

    def test_evaluate_summary(self, evaluator: BioASQEvaluator) -> None:
        """Test summary evaluation."""
        question = BioASQQuestion(
            question_id="q1",
            question_text="Describe aspirin's mechanism.",
            question_type="summary",
            ideal_answer="Aspirin inhibits cyclooxygenase enzymes, reducing prostaglandin synthesis.",
        )
        prediction = EvalPrediction(
            question_id="q1",
            predicted_answer="Aspirin works by inhibiting cyclooxygenase, which reduces prostaglandins.",
        )

        result = evaluator._evaluate_summary(question, prediction)
        assert result["rouge_l"] > 0.5  # Should have high similarity

    def test_evaluate_full(self, evaluator: BioASQEvaluator) -> None:
        """Test full evaluation across multiple questions."""
        questions = [
            BioASQQuestion(
                question_id="q1",
                question_text="Is aspirin an NSAID?",
                question_type="yesno",
                exact_answer="yes",
                gold_pmids=["123"],
            ),
            BioASQQuestion(
                question_id="q2",
                question_text="What is the capital of France?",
                question_type="factoid",
                exact_answer="Paris",
            ),
        ]
        predictions = [
            EvalPrediction(
                question_id="q1",
                predicted_label="yes",
                predicted_answer="Yes",
                retrieved_pmids=["123", "456"],
            ),
            EvalPrediction(
                question_id="q2",
                predicted_answer="Paris",
                retrieved_pmids=["789"],
            ),
        ]

        metrics = evaluator.evaluate(questions, predictions)

        assert metrics.num_questions == 2
        assert metrics.yesno_count == 1
        assert metrics.factoid_count == 1
        assert metrics.yesno_accuracy == 1.0
        assert metrics.factoid_em == 1.0


class TestPubMedQAEvaluator:
    """Tests for PubMedQA evaluator."""

    @pytest.fixture
    def evaluator(self) -> PubMedQAEvaluator:
        """Create evaluator instance."""
        return PubMedQAEvaluator(include_maybe=True)

    def test_extract_label_from_label_field(self, evaluator: PubMedQAEvaluator) -> None:
        """Test label extraction from predicted_label field."""
        pred = EvalPrediction(
            question_id="q1",
            predicted_label="yes",
            predicted_answer="",
        )
        assert evaluator._extract_label(pred) == "yes"

    def test_extract_label_from_answer(self, evaluator: PubMedQAEvaluator) -> None:
        """Test label extraction from answer text."""
        pred = EvalPrediction(
            question_id="q1",
            predicted_answer="Yes, the treatment is effective.",
        )
        assert evaluator._extract_label(pred) == "yes"

    def test_extract_label_no(self, evaluator: PubMedQAEvaluator) -> None:
        """Test label extraction for 'no' answer."""
        pred = EvalPrediction(
            question_id="q1",
            predicted_answer="No, there is no evidence.",
        )
        assert evaluator._extract_label(pred) == "no"

    def test_extract_label_maybe(self, evaluator: PubMedQAEvaluator) -> None:
        """Test label extraction for 'maybe' answer."""
        pred = EvalPrediction(
            question_id="q1",
            predicted_answer="It is uncertain whether this works.",
        )
        assert evaluator._extract_label(pred) == "maybe"

    def test_evaluate_single_correct(self, evaluator: PubMedQAEvaluator) -> None:
        """Test single question evaluation with correct answer."""
        question = PubMedQAQuestion(
            question_id="q1",
            question_text="Does aspirin reduce pain?",
            label="yes",
            pmid="12345",
        )
        prediction = EvalPrediction(
            question_id="q1",
            predicted_label="yes",
            predicted_answer="Yes",
        )

        result = evaluator.evaluate_single(question, prediction)
        assert result["correct"]
        assert result["gold"] == "yes"
        assert result["predicted"] == "yes"

    def test_evaluate_full(self, evaluator: PubMedQAEvaluator) -> None:
        """Test full evaluation."""
        questions = [
            PubMedQAQuestion(
                question_id="q1",
                question_text="Q1?",
                label="yes",
                pmid="123",
            ),
            PubMedQAQuestion(
                question_id="q2",
                question_text="Q2?",
                label="no",
                pmid="456",
            ),
            PubMedQAQuestion(
                question_id="q3",
                question_text="Q3?",
                label="maybe",
                pmid="789",
            ),
        ]
        predictions = [
            EvalPrediction(question_id="q1", predicted_label="yes"),
            EvalPrediction(question_id="q2", predicted_label="no"),
            EvalPrediction(question_id="q3", predicted_label="yes"),  # Wrong
        ]

        metrics = evaluator.evaluate(questions, predictions)

        assert metrics.num_questions == 3
        assert metrics.accuracy == pytest.approx(2 / 3)

    def test_evaluate_without_maybe(self) -> None:
        """Test evaluation without 'maybe' label."""
        evaluator = PubMedQAEvaluator(include_maybe=False)

        pred = EvalPrediction(
            question_id="q1",
            predicted_answer="It is uncertain.",
        )
        # Without maybe, uncertain maps to no
        assert evaluator._extract_label(pred) == "no"


class TestMetricResults:
    """Tests for metric result conversion."""

    def test_bioasq_to_metric_results(self) -> None:
        """Test BioASQMetrics to MetricResult conversion."""
        metrics = BioASQMetrics(
            num_questions=100,
            yesno_count=25,
            factoid_count=25,
            list_count=25,
            summary_count=25,
            yesno_accuracy=0.8,
            factoid_em=0.7,
            factoid_f1=0.75,
            list_f1=0.65,
            summary_rouge_l=0.6,
            recall_at_10=0.9,
            mrr=0.85,
        )

        results = metrics.to_metric_results()

        assert "yesno_accuracy" in results
        assert results["yesno_accuracy"].value == 0.8
        assert results["yesno_accuracy"].count == 25

    def test_pubmedqa_to_metric_results(self) -> None:
        """Test PubMedQAMetrics to MetricResult conversion."""
        metrics = PubMedQAMetrics(
            num_questions=100,
            accuracy=0.75,
            macro_f1=0.72,
            yes_f1=0.8,
            no_f1=0.7,
            maybe_f1=0.66,
        )

        results = metrics.to_metric_results()

        assert "accuracy" in results
        assert results["accuracy"].value == 0.75
        assert "macro_f1" in results
        assert results["macro_f1"].value == 0.72

