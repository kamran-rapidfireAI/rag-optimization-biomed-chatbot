"""Unit tests for experiment runner and sweep functionality."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from biorag.experiments.artifacts import (
    ArtifactManager,
    ReproducibilityInfo,
    get_git_branch,
    get_git_commit,
    get_git_dirty,
)
from biorag.experiments.runner import ExperimentRunner, RunLatency, RunResult
from biorag.experiments.sweep import (
    SweepRunner,
    create_parameter,
    generate_grid,
    _set_nested,
    _get_nested,
)
from biorag.schemas.config import BioRAGConfig
from biorag.schemas.evaluation import EvalResult, RunMetrics
from biorag.schemas.experiments import (
    ParameterRange,
    SweepConfig,
    SweepParameter,
    SweepResult,
)


class TestParameterRange:
    """Tests for ParameterRange schema."""

    def test_grid_values(self) -> None:
        """Test grid values extraction."""
        param_range = ParameterRange(type="grid", values=[1, 2, 3])
        assert param_range.get_values() == [1, 2, 3]

    def test_choice_values(self) -> None:
        """Test choice values extraction."""
        param_range = ParameterRange(type="choice", values=["a", "b", "c"])
        assert param_range.get_values() == ["a", "b", "c"]

    def test_range_with_step(self) -> None:
        """Test range with step size."""
        param_range = ParameterRange(type="range", min=0.0, max=1.0, step=0.25)
        values = param_range.get_values()
        assert len(values) == 5
        assert values[0] == 0.0
        assert values[-1] == 1.0

    def test_range_with_num(self) -> None:
        """Test range with number of values."""
        param_range = ParameterRange(type="range", min=0.0, max=1.0, num=5)
        values = param_range.get_values()
        assert len(values) == 5
        assert values[0] == 0.0
        assert values[-1] == 1.0

    def test_range_single_value(self) -> None:
        """Test range with single value."""
        param_range = ParameterRange(type="range", min=0.5, max=0.5, num=1)
        assert param_range.get_values() == [0.5]

    def test_empty_range(self) -> None:
        """Test empty parameter range."""
        param_range = ParameterRange(type="grid")
        assert param_range.get_values() == []


class TestSweepParameter:
    """Tests for SweepParameter schema."""

    def test_get_values(self) -> None:
        """Test getting values from parameter."""
        param = SweepParameter(
            path="retrieval.k",
            range=ParameterRange(type="grid", values=[5, 10, 15]),
        )
        assert param.get_values() == [5, 10, 15]

    def test_path_format(self) -> None:
        """Test parameter path format."""
        param = SweepParameter(
            path="chunking.chunk_size",
            range=ParameterRange(type="grid", values=[100]),
        )
        assert param.path == "chunking.chunk_size"


class TestSweepConfig:
    """Tests for SweepConfig schema."""

    def test_get_num_configs_single_param(self) -> None:
        """Test counting configs with single parameter."""
        config = SweepConfig(
            name="test",
            parameters=[
                SweepParameter(
                    path="retrieval.k",
                    range=ParameterRange(type="grid", values=[5, 10, 15]),
                )
            ],
        )
        assert config.get_num_configs() == 3

    def test_get_num_configs_multiple_params(self) -> None:
        """Test counting configs with multiple parameters (cartesian product)."""
        config = SweepConfig(
            name="test",
            parameters=[
                SweepParameter(
                    path="retrieval.k",
                    range=ParameterRange(type="grid", values=[5, 10]),
                ),
                SweepParameter(
                    path="retrieval.mode",
                    range=ParameterRange(type="choice", values=["similarity", "mmr"]),
                ),
            ],
        )
        assert config.get_num_configs() == 4  # 2 * 2

    def test_get_num_configs_empty(self) -> None:
        """Test counting with no parameters."""
        config = SweepConfig(name="test", parameters=[])
        assert config.get_num_configs() == 1

    def test_from_yaml(self, temp_dir: Path) -> None:
        """Test loading from YAML."""
        yaml_content = """
name: test_sweep
description: Test sweep
parameters:
  - path: retrieval.k
    range:
      type: grid
      values: [5, 10]
dataset: bioasq
max_questions: 10
"""
        yaml_path = temp_dir / "sweep.yaml"
        yaml_path.write_text(yaml_content)

        config = SweepConfig.from_yaml(str(yaml_path))
        assert config.name == "test_sweep"
        assert len(config.parameters) == 1
        assert config.max_questions == 10

    def test_to_yaml(self, temp_dir: Path) -> None:
        """Test saving to YAML."""
        config = SweepConfig(
            name="test",
            parameters=[
                SweepParameter(
                    path="retrieval.k",
                    range=ParameterRange(type="grid", values=[5]),
                )
            ],
        )
        yaml_path = temp_dir / "output.yaml"
        config.to_yaml(str(yaml_path))
        assert yaml_path.exists()


class TestGenerateGrid:
    """Tests for grid generation function."""

    def test_empty_parameters(self) -> None:
        """Test with no parameters."""
        configs = generate_grid([])
        assert configs == [{}]

    def test_single_parameter(self) -> None:
        """Test with single parameter."""
        params = [
            SweepParameter(
                path="retrieval.k",
                range=ParameterRange(type="grid", values=[5, 10, 15]),
            )
        ]
        configs = generate_grid(params)
        assert len(configs) == 3
        assert configs[0] == {"retrieval": {"k": 5}}
        assert configs[1] == {"retrieval": {"k": 10}}
        assert configs[2] == {"retrieval": {"k": 15}}

    def test_multiple_parameters(self) -> None:
        """Test cartesian product of multiple parameters."""
        params = [
            SweepParameter(
                path="retrieval.k",
                range=ParameterRange(type="grid", values=[5, 10]),
            ),
            SweepParameter(
                path="chunking.chunk_size",
                range=ParameterRange(type="grid", values=[200, 400]),
            ),
        ]
        configs = generate_grid(params)
        assert len(configs) == 4  # 2 * 2

        # Check all combinations exist
        expected = [
            {"retrieval": {"k": 5}, "chunking": {"chunk_size": 200}},
            {"retrieval": {"k": 5}, "chunking": {"chunk_size": 400}},
            {"retrieval": {"k": 10}, "chunking": {"chunk_size": 200}},
            {"retrieval": {"k": 10}, "chunking": {"chunk_size": 400}},
        ]
        assert len(configs) == len(expected)

    def test_nested_path(self) -> None:
        """Test deeply nested path."""
        params = [
            SweepParameter(
                path="llm.temperature",
                range=ParameterRange(type="grid", values=[0.0, 0.5]),
            )
        ]
        configs = generate_grid(params)
        assert configs[0] == {"llm": {"temperature": 0.0}}
        assert configs[1] == {"llm": {"temperature": 0.5}}


class TestNestedDictHelpers:
    """Tests for nested dict helper functions."""

    def test_set_nested_simple(self) -> None:
        """Test setting simple nested value."""
        d: dict = {}
        _set_nested(d, "a.b.c", 42)
        assert d == {"a": {"b": {"c": 42}}}

    def test_set_nested_existing(self) -> None:
        """Test setting value in existing structure."""
        d: dict = {"a": {"b": {"x": 1}}}
        _set_nested(d, "a.b.c", 42)
        assert d == {"a": {"b": {"x": 1, "c": 42}}}

    def test_get_nested_existing(self) -> None:
        """Test getting existing nested value."""
        d = {"a": {"b": {"c": 42}}}
        assert _get_nested(d, "a.b.c") == 42

    def test_get_nested_missing(self) -> None:
        """Test getting missing nested value."""
        d = {"a": {"b": {}}}
        assert _get_nested(d, "a.b.c") is None
        assert _get_nested(d, "a.b.c", "default") == "default"


class TestCreateParameter:
    """Tests for create_parameter helper function."""

    def test_create_with_values(self) -> None:
        """Test creating parameter with explicit values."""
        param = create_parameter("retrieval.k", values=[5, 10, 15])
        assert param.path == "retrieval.k"
        assert param.get_values() == [5, 10, 15]

    def test_create_with_range(self) -> None:
        """Test creating parameter with range."""
        param = create_parameter("temperature", min_val=0.0, max_val=1.0, num=3)
        assert param.path == "temperature"
        values = param.get_values()
        assert len(values) == 3

    def test_create_invalid(self) -> None:
        """Test creating parameter with missing args."""
        with pytest.raises(ValueError):
            create_parameter("test")


class TestRunLatency:
    """Tests for RunLatency dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        latency = RunLatency(
            pipeline_setup_ms=10.0,
            index_load_ms=20.0,
            evaluation_ms=100.0,
            total_ms=130.0,
        )
        d = latency.to_dict()
        assert d["pipeline_setup_ms"] == 10.0
        assert d["total_ms"] == 130.0


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_primary_metric_from_accuracy(self) -> None:
        """Test primary metric extraction from accuracy."""
        from biorag.schemas.evaluation import MetricResult

        metrics = RunMetrics(
            run_id="test",
            dataset="bioasq",
            num_questions=10,
            answer_metrics={
                "accuracy": MetricResult(name="accuracy", value=0.85, count=10)
            },
        )
        result = RunResult(
            run_id="test",
            config=BioRAGConfig(),
            eval_result=EvalResult(run_id="test", metrics=metrics),
            latency=RunLatency(),
        )
        assert result.primary_metric == 0.85

    def test_primary_metric_fallback(self) -> None:
        """Test primary metric fallback when no metrics."""
        result = RunResult(
            run_id="test",
            config=BioRAGConfig(),
            eval_result=EvalResult(run_id="test"),
            latency=RunLatency(),
        )
        assert result.primary_metric == 0.0

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        result = RunResult(
            run_id="test",
            config=BioRAGConfig(),
            eval_result=EvalResult(run_id="test"),
            latency=RunLatency(total_ms=100.0),
            status="completed",
        )
        d = result.to_dict()
        assert d["run_id"] == "test"
        assert d["status"] == "completed"
        assert d["latency"]["total_ms"] == 100.0


class TestReproducibilityInfo:
    """Tests for ReproducibilityInfo dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        info = ReproducibilityInfo(
            git_commit="abc123",
            git_branch="main",
            git_dirty=False,
            timestamp=datetime.utcnow().isoformat(),
            config={},
            model_versions={"llm": "gpt-4o-mini"},
            dataset="bioasq",
            dataset_version=None,
            random_seed=42,
            total_tokens=1000,
            estimated_cost_usd=0.01,
            cache_hit_rate=0.5,
        )
        d = info.to_dict()
        assert d["git_commit"] == "abc123"
        assert d["random_seed"] == 42


class TestGitHelpers:
    """Tests for git helper functions."""

    def test_get_git_commit_returns_string_or_none(self) -> None:
        """Test git commit retrieval."""
        result = get_git_commit()
        assert result is None or (isinstance(result, str) and len(result) == 8)

    def test_get_git_branch_returns_string_or_none(self) -> None:
        """Test git branch retrieval."""
        result = get_git_branch()
        assert result is None or isinstance(result, str)

    def test_get_git_dirty_returns_bool(self) -> None:
        """Test git dirty check."""
        result = get_git_dirty()
        assert isinstance(result, bool)


class TestArtifactManager:
    """Tests for ArtifactManager."""

    def test_init_creates_dir(self, temp_dir: Path) -> None:
        """Test initialization creates output directory."""
        manager = ArtifactManager(temp_dir / "artifacts")
        assert (temp_dir / "artifacts").exists()

    def test_list_runs_empty(self, temp_dir: Path) -> None:
        """Test listing runs in empty directory."""
        manager = ArtifactManager(temp_dir)
        assert manager.list_runs() == []

    def test_generate_leaderboard_empty(self, temp_dir: Path) -> None:
        """Test generating leaderboard with no results."""
        manager = ArtifactManager(temp_dir)
        path = manager.generate_leaderboard([])
        assert path.exists()

    def test_load_leaderboard_missing(self, temp_dir: Path) -> None:
        """Test loading missing leaderboard."""
        manager = ArtifactManager(temp_dir)
        assert manager.load_leaderboard() == []

    def test_save_and_load_run(self, temp_dir: Path) -> None:
        """Test saving and loading run."""
        manager = ArtifactManager(temp_dir)

        # Create mock result
        result = RunResult(
            run_id="test_run",
            config=BioRAGConfig(),
            eval_result=EvalResult(run_id="test_run"),
            latency=RunLatency(total_ms=100.0),
            status="completed",
        )

        # Save
        run_dir = manager.save_run(result, dataset="bioasq")
        assert (run_dir / "run.json").exists()
        assert (run_dir / "config.yaml").exists()

        # Load
        data = manager.load_run("test_run")
        assert data["run_id"] == "test_run"
        assert data["status"] == "completed"

        # List
        runs = manager.list_runs()
        assert "test_run" in runs

    def test_generate_leaderboard_with_results(self, temp_dir: Path) -> None:
        """Test generating leaderboard with results."""
        from biorag.schemas.evaluation import MetricResult

        manager = ArtifactManager(temp_dir)

        # Create mock results
        results = [
            RunResult(
                run_id="run_1",
                config=BioRAGConfig(),
                eval_result=EvalResult(
                    run_id="run_1",
                    metrics=RunMetrics(
                        run_id="run_1",
                        dataset="bioasq",
                        num_questions=10,
                        answer_metrics={
                            "accuracy": MetricResult(name="accuracy", value=0.9, count=10)
                        },
                    ),
                ),
                latency=RunLatency(total_ms=100.0),
            ),
            RunResult(
                run_id="run_2",
                config=BioRAGConfig(),
                eval_result=EvalResult(
                    run_id="run_2",
                    metrics=RunMetrics(
                        run_id="run_2",
                        dataset="bioasq",
                        num_questions=10,
                        answer_metrics={
                            "accuracy": MetricResult(name="accuracy", value=0.8, count=10)
                        },
                    ),
                ),
                latency=RunLatency(total_ms=150.0),
            ),
        ]

        path = manager.generate_leaderboard(results)
        assert path.exists()

        # Load and verify
        rows = manager.load_leaderboard(path)
        assert len(rows) == 2
        assert rows[0]["rank"] == "1"
        assert rows[0]["run_id"] == "run_1"  # Higher accuracy should be first


class TestSweepResult:
    """Tests for SweepResult schema."""

    def test_creation(self) -> None:
        """Test creating SweepResult."""
        result = SweepResult(
            sweep_name="test",
            sweep_config={},
            total_runs=10,
            completed_runs=9,
            failed_runs=1,
            best_run_id="run_5",
            best_metric=0.95,
            best_config={"retrieval": {"k": 10}},
            average_metric=0.85,
            total_cost_usd=0.50,
            leaderboard_path="/path/to/leaderboard.csv",
        )
        assert result.sweep_name == "test"
        assert result.completed_runs == 9


class TestExperimentRunnerMocked:
    """Tests for ExperimentRunner with mocked dependencies."""

    def test_apply_overrides(self, temp_dir: Path) -> None:
        """Test config override application."""
        runner = ExperimentRunner(
            base_config=BioRAGConfig(),
            output_dir=temp_dir,
        )

        overrides = {"retrieval": {"k": 20}}
        new_config = runner._apply_overrides(overrides)

        assert new_config.retrieval.k == 20
        # Base config should be unchanged
        assert runner.base_config.retrieval.k == 10

    def test_init_creates_directories(self, temp_dir: Path) -> None:
        """Test that initialization creates necessary directories."""
        runner = ExperimentRunner(
            base_config=BioRAGConfig(
                paths={"runs_dir": temp_dir / "runs", "cache_dir": temp_dir / "cache"}
            ),
            output_dir=temp_dir / "output",
        )
        assert (temp_dir / "output").exists()


class TestSweepRunnerMocked:
    """Tests for SweepRunner with mocked dependencies."""

    def test_init(self, temp_dir: Path) -> None:
        """Test SweepRunner initialization."""
        runner = SweepRunner(
            base_config=BioRAGConfig(
                paths={"runs_dir": temp_dir / "runs", "cache_dir": temp_dir / "cache"}
            ),
            output_dir=temp_dir / "sweeps",
        )
        assert (temp_dir / "sweeps").exists()
        assert runner.artifact_manager is not None


class TestRapidFireAdapterImports:
    """Tests for RapidFire AI adapter module."""

    def test_check_rapidfire_available(self) -> None:
        """Test the rapidfire availability check function."""
        from biorag.experiments.rapidfire_adapter import check_rapidfire_available

        # Should return a boolean
        result = check_rapidfire_available()
        assert isinstance(result, bool)

    def test_dataset_adapter_init(self) -> None:
        """Test BioRAGDatasetAdapter initialization without rapidfireai."""
        from biorag.experiments.rapidfire_adapter import BioRAGDatasetAdapter

        adapter = BioRAGDatasetAdapter()
        assert adapter.config is not None

    def test_dataset_adapter_with_config(self) -> None:
        """Test BioRAGDatasetAdapter with custom config."""
        from biorag.experiments.rapidfire_adapter import BioRAGDatasetAdapter

        config = BioRAGConfig()
        adapter = BioRAGDatasetAdapter(config)
        assert adapter.config == config


class TestRapidFireCallbackFunctions:
    """Tests for RapidFire AI callback functions (when rapidfireai is available)."""

    def test_callback_functions_created(self) -> None:
        """Test that callback functions can be created when rapidfireai available."""
        from biorag.experiments.rapidfire_adapter import (
            RAPIDFIRE_AVAILABLE,
            BioRAGRapidFireAdapter,
        )

        if not RAPIDFIRE_AVAILABLE:
            pytest.skip("rapidfireai not installed")

        adapter = BioRAGRapidFireAdapter()

        # These should return callable functions
        preprocess_fn = adapter.create_preprocess_fn()
        assert callable(preprocess_fn)

        postprocess_fn = adapter.create_postprocess_fn()
        assert callable(postprocess_fn)

        compute_metrics_fn = adapter.create_compute_metrics_fn()
        assert callable(compute_metrics_fn)

        accumulate_metrics_fn = adapter.create_accumulate_metrics_fn()
        assert callable(accumulate_metrics_fn)


class TestMetricsComputation:
    """Tests for metric computation functions."""

    def test_compute_metrics_fn_logic(self) -> None:
        """Test the compute_metrics_fn logic."""
        from biorag.experiments.rapidfire_adapter import (
            RAPIDFIRE_AVAILABLE,
            BioRAGRapidFireAdapter,
        )

        if not RAPIDFIRE_AVAILABLE:
            pytest.skip("rapidfireai not installed")

        adapter = BioRAGRapidFireAdapter()
        compute_metrics_fn = adapter.create_compute_metrics_fn()

        # Test with mock batch
        batch = {
            "query": ["question 1", "question 2"],
            "retrieved_documents": [["doc1", "doc2"], ["doc3"]],
            "gold_pmids": [["doc1"], ["doc4"]],
            "answer": ["yes", "no"],
            "expected": ["yes", "no"],
        }

        result = compute_metrics_fn(batch)

        assert "Total" in result
        assert result["Total"]["value"] == 2
        assert "Accuracy" in result
        assert result["Accuracy"]["value"] == 1.0  # Both answers match

    def test_accumulate_metrics_fn_logic(self) -> None:
        """Test the accumulate_metrics_fn logic."""
        from biorag.experiments.rapidfire_adapter import (
            RAPIDFIRE_AVAILABLE,
            BioRAGRapidFireAdapter,
        )

        if not RAPIDFIRE_AVAILABLE:
            pytest.skip("rapidfireai not installed")

        adapter = BioRAGRapidFireAdapter()
        accumulate_fn = adapter.create_accumulate_metrics_fn()

        # Test with aggregated metrics from multiple batches
        aggregated_metrics = {
            "Total": [{"value": 10}, {"value": 10}],
            "Accuracy": [{"value": 0.8}, {"value": 0.9}],
            "F1": [{"value": 0.75}, {"value": 0.85}],
        }

        result = accumulate_fn(aggregated_metrics)

        assert result["Total"]["value"] == 20
        # Weighted average: (0.8*10 + 0.9*10) / 20 = 0.85
        assert abs(result["Accuracy"]["value"] - 0.85) < 0.001


