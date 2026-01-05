"""Integration tests for experiment sweep functionality."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from biorag.experiments.runner import ExperimentRunner, RunResult, RunLatency
from biorag.experiments.sweep import SweepRunner, generate_grid, create_parameter
from biorag.experiments.artifacts import ArtifactManager
from biorag.pipeline.rag import RAGResult, PipelineLatency, PipelineDebugInfo
from biorag.schemas.config import BioRAGConfig
from biorag.schemas.evaluation import (
    BioASQQuestion,
    EvalPrediction,
    EvalResult,
    MetricResult,
    RetrievalResult,
    RunMetrics,
)
from biorag.schemas.experiments import SweepConfig, SweepParameter, ParameterRange
from biorag.schemas.generation import AnswerOutput, GenerationResponse


@pytest.fixture
def mock_rag_pipeline() -> MagicMock:
    """Create a mock RAG pipeline that returns consistent results."""
    pipeline = MagicMock()

    def mock_query(question: str, question_type: str | None = None) -> RAGResult:
        label = None
        if question_type == "yesno":
            label = "yes"

        return RAGResult(
            answer=AnswerOutput(
                answer="Test answer for the question.",
                label=label,
                abstained=False,
            ),
            retrieved_chunks=[
                RetrievalResult(
                    pmid="12345",
                    chunk_id="chunk_1",
                    text="Test chunk",
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
                    text="Test chunk",
                    score=0.9,
                    rank=1,
                    rerank_score=0.95,
                    rerank_rank=1,
                ),
            ],
            latency=PipelineLatency(
                retrieve_ms=10.0,
                rerank_ms=5.0,
                generate_ms=50.0,
                total_ms=65.0,
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
def mock_harness(mock_rag_pipeline: MagicMock) -> MagicMock:
    """Create a mock evaluation harness."""
    harness = MagicMock()
    harness.pipeline = mock_rag_pipeline

    # Mock load_golden_suite
    def load_golden_suite(
        dataset: str,
        split: str = "train",
        max_questions: int | None = None,
        seed: int = 42,
    ):
        questions = [
            BioASQQuestion(
                question_id=f"q{i}",
                question_text=f"Test question {i}?",
                question_type="yesno",
                exact_answer="yes",
            )
            for i in range(max_questions or 5)
        ]
        return questions

    harness.load_golden_suite = load_golden_suite

    # Mock evaluate_bioasq
    def evaluate_bioasq(questions, run_id=None, **kwargs):
        return EvalResult(
            run_id=run_id or "test",
            metrics=RunMetrics(
                run_id=run_id or "test",
                dataset="bioasq",
                num_questions=len(questions),
                answer_metrics={
                    "accuracy": MetricResult(name="accuracy", value=0.8, count=len(questions)),
                    "token_f1": MetricResult(name="token_f1", value=0.75, count=len(questions)),
                },
                estimated_cost_usd=0.01,
            ),
        )

    harness.evaluate_bioasq = evaluate_bioasq
    harness.evaluate_pubmedqa = evaluate_bioasq  # Use same mock

    return harness


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


class TestGridGeneration:
    """Tests for grid generation with real parameters."""

    def test_single_param_sweep(self) -> None:
        """Test grid with single parameter."""
        params = [
            create_parameter("retrieval.k", values=[5, 10, 15, 20]),
        ]
        configs = generate_grid(params)

        assert len(configs) == 4
        assert all("retrieval" in c for c in configs)

    def test_multi_param_sweep(self) -> None:
        """Test grid with multiple parameters produces cartesian product."""
        params = [
            create_parameter("retrieval.k", values=[5, 10]),
            create_parameter("retrieval.mode", values=["similarity", "mmr"]),
            create_parameter("chunking.chunk_size", values=[200, 400]),
        ]
        configs = generate_grid(params)

        assert len(configs) == 8  # 2 * 2 * 2

    def test_range_param_sweep(self) -> None:
        """Test grid with range parameter."""
        params = [
            create_parameter("llm.temperature", min_val=0.0, max_val=1.0, num=5),
        ]
        configs = generate_grid(params)

        assert len(configs) == 5
        temps = [c["llm"]["temperature"] for c in configs]
        assert temps[0] == 0.0
        assert temps[-1] == 1.0


class TestExperimentRunner:
    """Integration tests for ExperimentRunner."""

    def test_runner_initialization(self, test_config: BioRAGConfig, temp_dir: Path) -> None:
        """Test runner initializes correctly."""
        runner = ExperimentRunner(
            base_config=test_config,
            output_dir=temp_dir / "runs",
        )

        assert runner.base_config == test_config
        assert runner.output_dir == temp_dir / "runs"

    @patch("biorag.experiments.runner.EvaluationHarness")
    def test_run_with_overrides(
        self,
        mock_harness_class: MagicMock,
        test_config: BioRAGConfig,
        temp_dir: Path,
        mock_harness: MagicMock,
    ) -> None:
        """Test running with config overrides."""
        mock_harness_class.return_value = mock_harness

        runner = ExperimentRunner(
            base_config=test_config,
            output_dir=temp_dir / "runs",
        )

        result = runner.run(
            config_overrides={"retrieval": {"k": 20}},
            max_questions=3,
            save_artifacts=False,
        )

        assert result.status == "completed"
        assert result.config.retrieval.k == 20
        assert result.latency.total_ms > 0


class TestSweepRunner:
    """Integration tests for SweepRunner."""

    def test_sweep_runner_initialization(
        self, test_config: BioRAGConfig, temp_dir: Path
    ) -> None:
        """Test sweep runner initializes correctly."""
        runner = SweepRunner(
            base_config=test_config,
            output_dir=temp_dir / "sweeps",
        )

        assert runner.output_dir == temp_dir / "sweeps"
        assert runner.artifact_manager is not None

    @patch("biorag.experiments.runner.EvaluationHarness")
    def test_mini_sweep(
        self,
        mock_harness_class: MagicMock,
        test_config: BioRAGConfig,
        temp_dir: Path,
        mock_harness: MagicMock,
    ) -> None:
        """Test running a minimal sweep (2 configs)."""
        mock_harness_class.return_value = mock_harness

        runner = SweepRunner(
            base_config=test_config,
            output_dir=temp_dir / "sweeps",
        )

        sweep_config = SweepConfig(
            name="test_mini",
            parameters=[
                SweepParameter(
                    path="retrieval.k",
                    range=ParameterRange(type="grid", values=[5, 10]),
                ),
            ],
            max_questions=3,
            save_artifacts=True,
        )

        result = runner.run_sweep(sweep_config)

        assert result.sweep_name == "test_mini"
        assert result.total_runs == 2
        assert result.completed_runs == 2
        assert result.failed_runs == 0
        assert result.best_run_id is not None
        assert result.leaderboard_path is not None

        # Verify leaderboard was created
        assert Path(result.leaderboard_path).exists()

    @patch("biorag.experiments.runner.EvaluationHarness")
    def test_sweep_with_progress_callback(
        self,
        mock_harness_class: MagicMock,
        test_config: BioRAGConfig,
        temp_dir: Path,
        mock_harness: MagicMock,
    ) -> None:
        """Test sweep with progress callback."""
        mock_harness_class.return_value = mock_harness

        runner = SweepRunner(
            base_config=test_config,
            output_dir=temp_dir / "sweeps",
        )

        progress_calls: list[tuple[int, int]] = []

        def track_progress(current: int, total: int, result: RunResult | None) -> None:
            progress_calls.append((current, total))

        sweep_config = SweepConfig(
            name="test_progress",
            parameters=[
                SweepParameter(
                    path="retrieval.k",
                    range=ParameterRange(type="grid", values=[5, 10, 15]),
                ),
            ],
            max_questions=2,
            save_artifacts=False,
        )

        runner.run_sweep(sweep_config, progress_callback=track_progress)

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)


class TestSweepConfigLoading:
    """Tests for loading sweep configs from YAML files."""

    def test_load_quick_test_config(self, temp_dir: Path) -> None:
        """Test loading the quick_test sweep config."""
        config_content = """
name: quick_test
description: "Minimal sweep for testing"
parameters:
  - path: retrieval.k
    range:
      type: grid
      values: [5, 10]
dataset: bioasq
max_questions: 5
seed: 42
save_artifacts: true
"""
        config_path = temp_dir / "quick_test.yaml"
        config_path.write_text(config_content)

        config = SweepConfig.from_yaml(str(config_path))

        assert config.name == "quick_test"
        assert config.get_num_configs() == 2
        assert config.max_questions == 5


class TestArtifactIntegration:
    """Integration tests for artifact management."""

    @patch("biorag.experiments.runner.EvaluationHarness")
    def test_artifacts_saved_correctly(
        self,
        mock_harness_class: MagicMock,
        test_config: BioRAGConfig,
        temp_dir: Path,
        mock_harness: MagicMock,
    ) -> None:
        """Test that artifacts are saved correctly during sweep."""
        mock_harness_class.return_value = mock_harness

        runner = SweepRunner(
            base_config=test_config,
            output_dir=temp_dir / "sweeps",
        )

        sweep_config = SweepConfig(
            name="artifact_test",
            parameters=[
                SweepParameter(
                    path="retrieval.k",
                    range=ParameterRange(type="grid", values=[5]),
                ),
            ],
            max_questions=2,
            save_artifacts=True,
        )

        result = runner.run_sweep(sweep_config)

        # Check sweep directory was created
        sweep_dirs = list((temp_dir / "sweeps").glob("sweep_artifact_test_*"))
        assert len(sweep_dirs) == 1

        sweep_dir = sweep_dirs[0]

        # Check leaderboard
        assert (sweep_dir / "leaderboard.csv").exists()

        # Check sweep summary
        assert (sweep_dir / "sweep_summary.json").exists()

    def test_leaderboard_generation(self, temp_dir: Path) -> None:
        """Test leaderboard generation with multiple results."""
        manager = ArtifactManager(temp_dir)

        # Create results with different metrics
        results = []
        for i in range(5):
            metric_value = 0.7 + (i * 0.05)  # 0.7, 0.75, 0.8, 0.85, 0.9
            result = RunResult(
                run_id=f"run_{i}",
                config=BioRAGConfig(),
                eval_result=EvalResult(
                    run_id=f"run_{i}",
                    metrics=RunMetrics(
                        run_id=f"run_{i}",
                        dataset="bioasq",
                        num_questions=10,
                        answer_metrics={
                            "accuracy": MetricResult(
                                name="accuracy",
                                value=metric_value,
                                count=10,
                            )
                        },
                    ),
                ),
                latency=RunLatency(total_ms=100.0 + i * 10),
            )
            results.append(result)

        leaderboard_path = manager.generate_leaderboard(results)

        # Load and verify order
        rows = manager.load_leaderboard(leaderboard_path)
        assert len(rows) == 5

        # Should be sorted by metric descending
        assert rows[0]["run_id"] == "run_4"  # Highest metric
        assert rows[-1]["run_id"] == "run_0"  # Lowest metric

        # Ranks should be correct
        assert rows[0]["rank"] == "1"
        assert rows[-1]["rank"] == "5"





