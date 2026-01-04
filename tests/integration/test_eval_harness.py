"""Integration tests for evaluation harness."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from biorag.eval.harness import EvaluationHarness, EvalProgress, get_git_commit
from biorag.pipeline.rag import RAGResult, PipelineLatency, PipelineDebugInfo
from biorag.schemas.config import BioRAGConfig
from biorag.schemas.evaluation import (
    BioASQQuestion,
    EvalPrediction,
    PubMedQAQuestion,
    RetrievalResult,
)
from biorag.schemas.generation import AnswerOutput, GenerationResponse


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Create a mock RAG pipeline."""
    pipeline = MagicMock()
    
    # Mock query method
    def mock_query(question: str, question_type: str | None = None) -> RAGResult:
        # Return appropriate label based on question content
        label = None
        if question_type == "yesno":
            label = "yes"
        
        return RAGResult(
            answer=AnswerOutput(
                answer="This is a test answer.",
                label=label,
                abstained=False,
            ),
            retrieved_chunks=[
                RetrievalResult(
                    pmid="12345",
                    chunk_id="chunk_1",
                    text="Test chunk content",
                    score=0.9,
                    rank=1,
                    rerank_score=0.95,
                    rerank_rank=1,
                ),
            ],
            reranked_chunks=[
                RetrievalResult(
                    pmid="12345",
                    chunk_id="chunk_1",
                    text="Test chunk content",
                    score=0.9,
                    rank=1,
                    rerank_score=0.95,
                    rerank_rank=1,
                ),
            ],
            latency=PipelineLatency(
                retrieve_ms=10.0,
                rerank_ms=5.0,
                generate_ms=100.0,
                total_ms=115.0,
            ),
            debug_info=PipelineDebugInfo(),
            generation_response=GenerationResponse(
                answer=AnswerOutput(answer="Test", label=label),
                input_tokens=100,
                output_tokens=50,
            ),
        )
    
    pipeline.query = mock_query
    return pipeline


@pytest.fixture
def test_config(temp_dir: Path) -> BioRAGConfig:
    """Create test configuration."""
    return BioRAGConfig(
        paths={
            "data_dir": temp_dir / "data",
            "runs_dir": temp_dir / "runs",
            "cache_dir": temp_dir / "cache",
        },
    )


@pytest.fixture
def sample_bioasq_questions() -> list[BioASQQuestion]:
    """Create sample BioASQ questions."""
    return [
        BioASQQuestion(
            question_id="q1",
            question_text="Is aspirin an NSAID?",
            question_type="yesno",
            exact_answer="yes",
            gold_pmids=["12345"],
        ),
        BioASQQuestion(
            question_id="q2",
            question_text="What is the capital of France?",
            question_type="factoid",
            exact_answer="Paris",
            gold_pmids=["67890"],
        ),
        BioASQQuestion(
            question_id="q3",
            question_text="List three symptoms of flu.",
            question_type="list",
            exact_answer=["fever", "cough", "fatigue"],
        ),
    ]


@pytest.fixture
def sample_pubmedqa_questions() -> list[PubMedQAQuestion]:
    """Create sample PubMedQA questions."""
    return [
        PubMedQAQuestion(
            question_id="pq1",
            question_text="Does aspirin reduce inflammation?",
            label="yes",
            pmid="11111",
        ),
        PubMedQAQuestion(
            question_id="pq2",
            question_text="Is water flammable?",
            label="no",
            pmid="22222",
        ),
    ]


class TestEvalProgress:
    """Tests for EvalProgress tracking."""

    def test_progress_percentage(self) -> None:
        """Test progress percentage calculation."""
        progress = EvalProgress(total=100, completed=25)
        assert progress.progress_pct == 25.0

    def test_progress_percentage_zero_total(self) -> None:
        """Test progress with zero total."""
        progress = EvalProgress(total=0, completed=0)
        assert progress.progress_pct == 0.0

    def test_elapsed_seconds(self) -> None:
        """Test elapsed time calculation."""
        progress = EvalProgress(total=10)
        # Just check it returns a positive number
        assert progress.elapsed_seconds >= 0


class TestGetGitCommit:
    """Tests for git commit retrieval."""

    def test_get_git_commit_returns_string_or_none(self) -> None:
        """Test that get_git_commit returns a string or None."""
        result = get_git_commit()
        assert result is None or isinstance(result, str)
        if result:
            assert len(result) == 8  # Short SHA


class TestEvaluationHarness:
    """Tests for EvaluationHarness."""

    def test_init_with_config(self, test_config: BioRAGConfig, temp_dir: Path) -> None:
        """Test harness initialization with config."""
        harness = EvaluationHarness(
            config=test_config,
            output_dir=temp_dir / "output",
        )
        assert harness.config == test_config
        assert harness.output_dir == temp_dir / "output"

    def test_result_to_prediction(
        self,
        mock_pipeline: MagicMock,
        test_config: BioRAGConfig,
    ) -> None:
        """Test conversion of RAGResult to EvalPrediction."""
        harness = EvaluationHarness(
            pipeline=mock_pipeline,
            config=test_config,
        )

        result = mock_pipeline.query("Test question", question_type="yesno")
        prediction = harness._result_to_prediction("q1", result)

        assert prediction.question_id == "q1"
        assert prediction.predicted_answer == "This is a test answer."
        assert len(prediction.retrieved_pmids) == 1
        assert prediction.retrieved_pmids[0] == "12345"
        assert prediction.total_latency_ms == 115.0

    def test_run_predictions(
        self,
        mock_pipeline: MagicMock,
        test_config: BioRAGConfig,
    ) -> None:
        """Test running predictions on a batch of questions."""
        harness = EvaluationHarness(
            pipeline=mock_pipeline,
            config=test_config,
        )

        predictions = harness._run_predictions(
            questions=["Q1?", "Q2?"],
            question_ids=["q1", "q2"],
            question_types=["yesno", "factoid"],
        )

        assert len(predictions) == 2
        assert predictions[0].question_id == "q1"
        assert predictions[1].question_id == "q2"

    def test_evaluate_bioasq(
        self,
        mock_pipeline: MagicMock,
        test_config: BioRAGConfig,
        sample_bioasq_questions: list[BioASQQuestion],
        temp_dir: Path,
    ) -> None:
        """Test BioASQ evaluation."""
        harness = EvaluationHarness(
            pipeline=mock_pipeline,
            config=test_config,
            output_dir=temp_dir / "runs",
        )

        result = harness.evaluate_bioasq(
            sample_bioasq_questions,
            run_id="test_run",
            save_results=True,
        )

        assert result.run_id == "test_run"
        assert len(result.predictions) == 3
        assert result.metrics is not None
        assert result.metrics.num_questions == 3

        # Check files were saved
        run_dir = temp_dir / "runs" / "test_run"
        assert (run_dir / "run.json").exists()
        assert (run_dir / "predictions.jsonl").exists()
        assert (run_dir / "metrics.json").exists()

    def test_evaluate_pubmedqa(
        self,
        mock_pipeline: MagicMock,
        test_config: BioRAGConfig,
        sample_pubmedqa_questions: list[PubMedQAQuestion],
        temp_dir: Path,
    ) -> None:
        """Test PubMedQA evaluation."""
        harness = EvaluationHarness(
            pipeline=mock_pipeline,
            config=test_config,
            output_dir=temp_dir / "runs",
        )

        result = harness.evaluate_pubmedqa(
            sample_pubmedqa_questions,
            run_id="test_pubmedqa",
            save_results=False,
        )

        assert result.run_id == "test_pubmedqa"
        assert len(result.predictions) == 2
        assert result.metrics is not None

    def test_progress_callback(
        self,
        mock_pipeline: MagicMock,
        test_config: BioRAGConfig,
        sample_bioasq_questions: list[BioASQQuestion],
    ) -> None:
        """Test progress callback is called."""
        harness = EvaluationHarness(
            pipeline=mock_pipeline,
            config=test_config,
        )

        progress_updates: list[EvalProgress] = []

        def track_progress(progress: EvalProgress) -> None:
            progress_updates.append(progress)

        harness.evaluate_bioasq(
            sample_bioasq_questions,
            save_results=False,
            progress_callback=track_progress,
        )

        assert len(progress_updates) == 3  # One per question
        assert progress_updates[-1].completed == 3

    def test_build_eval_result(
        self,
        mock_pipeline: MagicMock,
        test_config: BioRAGConfig,
    ) -> None:
        """Test building complete EvalResult."""
        harness = EvaluationHarness(
            pipeline=mock_pipeline,
            config=test_config,
        )

        predictions = [
            EvalPrediction(
                question_id="q1",
                predicted_answer="Answer 1",
                retrieval_latency_ms=10.0,
                rerank_latency_ms=5.0,
                generation_latency_ms=100.0,
                total_latency_ms=115.0,
                input_tokens=100,
                output_tokens=50,
            ),
        ]

        from biorag.schemas.evaluation import MetricResult

        result = harness._build_eval_result(
            run_id="test",
            dataset="bioasq",
            predictions=predictions,
            retrieval_metrics={"recall@10": MetricResult(name="recall@10", value=0.8, count=1)},
            answer_metrics={"accuracy": MetricResult(name="accuracy", value=0.9, count=1)},
            num_questions=1,
            num_abstained=0,
        )

        assert result.run_id == "test"
        assert result.metrics is not None
        assert result.metrics.avg_total_latency_ms == 115.0
        assert "llm" in result.model_versions

    def test_max_questions_limit(
        self,
        mock_pipeline: MagicMock,
        test_config: BioRAGConfig,
        sample_bioasq_questions: list[BioASQQuestion],
    ) -> None:
        """Test max_questions parameter limits evaluation."""
        harness = EvaluationHarness(
            pipeline=mock_pipeline,
            config=test_config,
        )

        result = harness.evaluate_bioasq(
            sample_bioasq_questions,
            max_questions=1,
            save_results=False,
        )

        assert len(result.predictions) == 1


class TestErrorHandling:
    """Tests for error handling in evaluation."""

    def test_missing_prediction(self, test_config: BioRAGConfig) -> None:
        """Test handling of missing predictions."""
        from biorag.eval.bioasq_eval import BioASQEvaluator

        evaluator = BioASQEvaluator()

        questions = [
            BioASQQuestion(
                question_id="q1",
                question_text="Test?",
                question_type="yesno",
                exact_answer="yes",
            ),
        ]
        predictions: list[EvalPrediction] = []  # Empty predictions

        # Should not raise, just log warning
        metrics = evaluator.evaluate(questions, predictions)
        # Metrics should reflect no predictions matched
        assert metrics.yesno_count == 0

    def test_pipeline_error_creates_empty_prediction(
        self,
        test_config: BioRAGConfig,
        sample_bioasq_questions: list[BioASQQuestion],
    ) -> None:
        """Test that pipeline errors create empty predictions."""
        # Create a pipeline that raises an error
        failing_pipeline = MagicMock()
        failing_pipeline.query.side_effect = Exception("Test error")

        harness = EvaluationHarness(
            pipeline=failing_pipeline,
            config=test_config,
        )

        result = harness.evaluate_bioasq(
            sample_bioasq_questions[:1],
            save_results=False,
        )

        # Should have one prediction with abstention due to error
        assert len(result.predictions) == 1
        assert result.predictions[0].abstained
        assert "Error" in (result.predictions[0].abstention_reason or "")

