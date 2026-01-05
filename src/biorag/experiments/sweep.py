"""Parameter sweep runner for hyperparameter optimization.

Generates grid of configurations and executes them with:
- RapidFire AI integration for hyperparallelized execution (preferred)
- Fallback to sequential/parallel execution when rapidfireai unavailable
- Shared LLM cache for efficiency
- Leaderboard generation
"""

from __future__ import annotations

import itertools
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from biorag.experiments.artifacts import ArtifactManager, save_sweep_summary
from biorag.experiments.runner import ExperimentRunner, RunResult
from biorag.schemas.config import BioRAGConfig, load_config
from biorag.schemas.experiments import SweepConfig, SweepParameter, SweepResult
from biorag.utils.caching import LLMCache
from biorag.utils.logging import get_logger

logger = get_logger(__name__)

# Check RapidFire AI availability
try:
    from biorag.experiments.rapidfire_adapter import (
        RAPIDFIRE_AVAILABLE,
        RapidFireSweepRunner,
    )
except ImportError:
    RAPIDFIRE_AVAILABLE = False
    RapidFireSweepRunner = None  # type: ignore


def generate_grid(parameters: list[SweepParameter]) -> list[dict[str, Any]]:
    """
    Generate a grid of configuration overrides from sweep parameters.

    Args:
        parameters: List of SweepParameter definitions

    Returns:
        List of config override dictionaries
    """
    if not parameters:
        return [{}]

    # Get all parameter paths and their values
    param_paths: list[str] = []
    param_values: list[list[Any]] = []

    for param in parameters:
        values = param.get_values()
        if values:
            param_paths.append(param.path)
            param_values.append(values)

    if not param_paths:
        return [{}]

    # Generate cartesian product
    configs: list[dict[str, Any]] = []

    for combo in itertools.product(*param_values):
        config_override: dict[str, Any] = {}

        for path, value in zip(param_paths, combo):
            # Convert dot path to nested dict
            _set_nested(config_override, path, value)

        configs.append(config_override)

    logger.info(f"Generated {len(configs)} configurations from {len(parameters)} parameters")
    return configs


def _set_nested(d: dict[str, Any], path: str, value: Any) -> None:
    """Set a value in a nested dictionary using dot notation path."""
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _get_nested(d: dict[str, Any], path: str, default: Any = None) -> Any:
    """Get a value from a nested dictionary using dot notation path."""
    keys = path.split(".")
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


class SweepRunner:
    """
    Runs parameter sweeps over configurations.

    Features:
    - RapidFire AI integration for hyperparallelized execution (preferred)
    - Fallback to sequential/parallel execution when rapidfireai unavailable
    - Grid search over multiple parameters
    - Shared LLM cache for cost efficiency
    - Automatic leaderboard generation
    - Resume/restart support via run tracking

    RapidFire AI provides:
    - Hyperparallelized execution (16-24x throughput improvement)
    - Shard-based scheduling for concurrent configuration comparison
    - Interactive control (stop, resume, clone-modify runs)
    - Automatic GPU utilization and rate limit management
    """

    def __init__(
        self,
        base_config: BioRAGConfig | None = None,
        base_config_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        cache: LLMCache | None = None,
        index_path: str | Path | None = None,
        use_rapidfire: bool = True,
    ) -> None:
        """
        Initialize sweep runner.

        Args:
            base_config: Base configuration to override
            base_config_path: Path to base config file
            output_dir: Directory for saving sweep artifacts
            cache: Shared LLM cache
            index_path: Path to pre-built FAISS index
            use_rapidfire: Whether to use RapidFire AI when available (default: True)
        """
        # Load base config
        if base_config is not None:
            self.base_config = base_config
        elif base_config_path is not None:
            self.base_config = load_config(base_config_path)
        else:
            self.base_config = load_config()

        # Output directory
        self.output_dir = Path(output_dir) if output_dir else self.base_config.paths.runs_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Shared cache
        self.cache = cache or LLMCache(
            cache_dir=self.base_config.paths.cache_dir / "llm_cache"
        )

        # Index path
        self.index_path = Path(index_path) if index_path else None

        # Artifact manager
        self.artifact_manager = ArtifactManager(self.output_dir)

        # RapidFire AI integration
        self.use_rapidfire = use_rapidfire and RAPIDFIRE_AVAILABLE
        if self.use_rapidfire:
            logger.info("RapidFire AI available - using hyperparallelized execution")
        else:
            if use_rapidfire and not RAPIDFIRE_AVAILABLE:
                logger.warning(
                    "RapidFire AI requested but not available. "
                    "Install with: pip install rapidfireai. "
                    "Falling back to sequential execution."
                )

        logger.info(f"SweepRunner initialized: output_dir={self.output_dir}")

    def run_sweep(
        self,
        sweep_config: SweepConfig,
        progress_callback: Callable[[int, int, RunResult | None], None] | None = None,
        num_shards: int = 4,
    ) -> SweepResult:
        """
        Run a parameter sweep.

        Uses RapidFire AI for hyperparallelized execution when available,
        otherwise falls back to sequential/parallel execution.

        Args:
            sweep_config: Sweep configuration
            progress_callback: Callback for progress updates (current, total, result)
            num_shards: Number of shards for RapidFire AI parallel processing

        Returns:
            SweepResult with aggregated results
        """
        logger.info(f"Starting sweep: {sweep_config.name}")
        start_time = time.time()

        # Generate configuration grid
        configs = generate_grid(sweep_config.parameters)
        total_configs = len(configs)

        logger.info(f"Running {total_configs} configurations")

        # Create sweep-specific output directory
        sweep_output_dir = self.output_dir / f"sweep_{sweep_config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sweep_output_dir.mkdir(parents=True, exist_ok=True)

        # Try RapidFire AI execution first
        if self.use_rapidfire and sweep_config.parallel:
            try:
                return self._run_rapidfire_sweep(
                    sweep_config=sweep_config,
                    sweep_output_dir=sweep_output_dir,
                    num_shards=num_shards,
                    start_time=start_time,
                )
            except Exception as e:
                logger.warning(
                    f"RapidFire AI sweep failed: {e}. Falling back to sequential execution."
                )

        # Fallback: Create experiment runner for sequential/parallel execution
        runner = ExperimentRunner(
            base_config=self.base_config,
            cache=self.cache,
            output_dir=sweep_output_dir,
            index_path=self.index_path,
        )

        # Run configurations
        if sweep_config.parallel and sweep_config.max_parallel > 1:
            results = self._run_parallel(
                runner=runner,
                configs=configs,
                sweep_config=sweep_config,
                progress_callback=progress_callback,
            )
        else:
            results = self._run_sequential(
                runner=runner,
                configs=configs,
                sweep_config=sweep_config,
                progress_callback=progress_callback,
            )

        # Generate leaderboard
        leaderboard_path = self.artifact_manager.generate_leaderboard(
            results=results,
            output_path=sweep_output_dir / "leaderboard.csv",
        )

        # Save sweep summary
        save_sweep_summary(
            results=results,
            output_path=sweep_output_dir / "sweep_summary.json",
            sweep_config=sweep_config.model_dump(mode="json"),
        )

        # Build result
        completed = [r for r in results if r.status == "completed"]
        failed = [r for r in results if r.status == "failed"]

        best = max(completed, key=lambda r: r.primary_metric) if completed else None

        elapsed_time = time.time() - start_time
        best_metric_str = f"{best.primary_metric:.4f}" if best else "N/A"
        logger.info(
            f"Sweep '{sweep_config.name}' completed in {elapsed_time:.1f}s: "
            f"{len(completed)}/{total_configs} successful, "
            f"best metric: {best_metric_str}"
        )

        return SweepResult(
            sweep_name=sweep_config.name,
            sweep_config=sweep_config.model_dump(mode="json"),
            total_runs=total_configs,
            completed_runs=len(completed),
            failed_runs=len(failed),
            best_run_id=best.run_id if best else None,
            best_metric=best.primary_metric if best else None,
            best_config=best.config.model_dump(mode="json") if best else None,
            average_metric=sum(r.primary_metric for r in completed) / len(completed) if completed else 0.0,
            total_cost_usd=sum(
                r.eval_result.metrics.estimated_cost_usd
                for r in completed
                if r.eval_result.metrics
            ),
            leaderboard_path=str(leaderboard_path),
        )

    def _run_rapidfire_sweep(
        self,
        sweep_config: SweepConfig,
        sweep_output_dir: Path,
        num_shards: int,
        start_time: float,
    ) -> SweepResult:
        """
        Run sweep using RapidFire AI's hyperparallel framework.

        This provides 16-24x throughput improvement over sequential execution
        through shard-based scheduling and automatic GPU optimization.

        Uses the proper RapidFire AI patterns:
        - RFLangChainRagSpec for RAG pipeline configuration
        - RFOpenAIAPIModelConfig with rag parameter
        - preprocess_fn, postprocess_fn, compute_metrics_fn, accumulate_metrics_fn callbacks
        - RFGridSearch for hyperparameter sweeps

        Reference: https://github.com/RapidFireAI/rapidfireai/tree/main/tutorial_notebooks/rag-contexteng
        """
        if not RAPIDFIRE_AVAILABLE or RapidFireSweepRunner is None:
            raise RuntimeError("RapidFire AI not available")

        logger.info("Using RapidFire AI for hyperparallelized sweep execution")

        # Convert sweep parameters to RapidFire format
        # Map BioRAG parameter paths to RapidFire parameter names
        sweep_params: dict[str, list[Any]] = {}
        param_mapping = {
            "chunking.chunk_size": "chunk_size",
            "retrieval.search_type": "search_type",
            "retrieval.rerank_top_k": "reranker_top_n",
            "llm.model": "model",
            "llm.temperature": "temperature",
        }

        for param in sweep_config.parameters:
            values = param.get_values()
            # Map to RapidFire parameter name if available
            rf_param_name = param_mapping.get(param.path, param.path.split(".")[-1])
            sweep_params[rf_param_name] = values

        # Create RapidFire sweep runner
        rf_runner = RapidFireSweepRunner(
            experiment_name=sweep_config.name,
            base_config=self.base_config,
            experiment_path=sweep_output_dir,
            use_gpu=True,  # Use GPU when available
        )

        # Run the sweep
        rf_results = rf_runner.run_sweep(
            sweep_params=sweep_params,
            dataset=sweep_config.dataset,
            split=sweep_config.split,
            max_questions=sweep_config.max_questions,
            num_shards=num_shards,
            num_actors=sweep_config.max_parallel if sweep_config.max_parallel > 0 else 2,
            seed=sweep_config.seed,
        )

        # Convert RapidFire results to SweepResult format
        total_runs = len(rf_results)
        completed_runs = 0
        failed_runs = 0

        # Find best configuration
        best_run_id = None
        best_metric = None
        best_config = None
        total_cost = 0.0
        metric_sum = 0.0

        for run_id, (aggregated, cumulative) in rf_results.items():
            # Check for status in the results
            status = aggregated.get("status", "COMPLETED")
            if status == "COMPLETED" or status not in ["FAILED", "ERROR"]:
                completed_runs += 1

                # Get the primary metric (try Accuracy first, then F1)
                metric_value = cumulative.get("Accuracy", cumulative.get("F1", {}))
                if isinstance(metric_value, dict):
                    metric = metric_value.get("value", 0.0)
                else:
                    metric = float(metric_value) if metric_value else 0.0

                metric_sum += metric

                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    best_run_id = str(run_id)
                    best_config = aggregated.get("config", {})

                # Accumulate cost if available
                cost = cumulative.get("cost_usd", 0.0)
                if isinstance(cost, dict):
                    cost = cost.get("value", 0.0)
                total_cost += float(cost) if cost else 0.0
            else:
                failed_runs += 1

        elapsed_time = time.time() - start_time

        best_metric_str = f"{best_metric:.4f}" if best_metric is not None else "N/A"
        logger.info(
            f"RapidFire sweep '{sweep_config.name}' completed in {elapsed_time:.1f}s: "
            f"{completed_runs}/{total_runs} successful, best metric: {best_metric_str}"
        )

        # Get leaderboard from RapidFire and save
        leaderboard_path = None
        try:
            leaderboard_df = rf_runner.get_results_dataframe(rf_results)
            leaderboard_path = sweep_output_dir / "leaderboard.csv"
            leaderboard_df.to_csv(leaderboard_path, index=False)
        except Exception as e:
            logger.warning(f"Could not save RapidFire leaderboard: {e}")

        return SweepResult(
            sweep_name=sweep_config.name,
            sweep_config=sweep_config.model_dump(mode="json"),
            total_runs=total_runs,
            completed_runs=completed_runs,
            failed_runs=failed_runs,
            best_run_id=best_run_id,
            best_metric=best_metric,
            best_config=best_config,
            average_metric=metric_sum / completed_runs if completed_runs > 0 else 0.0,
            total_cost_usd=total_cost,
            leaderboard_path=str(leaderboard_path) if leaderboard_path else None,
        )

    def _run_sequential(
        self,
        runner: ExperimentRunner,
        configs: list[dict[str, Any]],
        sweep_config: SweepConfig,
        progress_callback: Callable[[int, int, RunResult | None], None] | None = None,
    ) -> list[RunResult]:
        """Run configurations sequentially (fallback when RapidFire AI unavailable)."""
        results: list[RunResult] = []

        for i, config_override in enumerate(configs):
            logger.info(f"Running configuration {i + 1}/{len(configs)}")

            result = runner.run(
                config_overrides=config_override,
                dataset=sweep_config.dataset,
                split=sweep_config.split,
                max_questions=sweep_config.max_questions,
                seed=sweep_config.seed,
                save_artifacts=sweep_config.save_artifacts,
            )
            results.append(result)

            # Save individual run artifacts
            if sweep_config.save_artifacts:
                self.artifact_manager.save_run(result, dataset=sweep_config.dataset)

            if progress_callback:
                progress_callback(i + 1, len(configs), result)

        return results

    def _run_parallel(
        self,
        runner: ExperimentRunner,
        configs: list[dict[str, Any]],
        sweep_config: SweepConfig,
        progress_callback: Callable[[int, int, RunResult | None], None] | None = None,
    ) -> list[RunResult]:
        """Run configurations in parallel using ThreadPoolExecutor."""
        results: list[RunResult] = []
        completed_count = 0

        def run_single(config_override: dict[str, Any], index: int) -> RunResult:
            return runner.run(
                config_overrides=config_override,
                dataset=sweep_config.dataset,
                split=sweep_config.split,
                max_questions=sweep_config.max_questions,
                seed=sweep_config.seed,
                save_artifacts=sweep_config.save_artifacts,
            )

        with ThreadPoolExecutor(max_workers=sweep_config.max_parallel) as executor:
            futures = {
                executor.submit(run_single, config, i): i
                for i, config in enumerate(configs)
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed_count += 1

                # Save individual run artifacts
                if sweep_config.save_artifacts:
                    self.artifact_manager.save_run(result, dataset=sweep_config.dataset)

                if progress_callback:
                    progress_callback(completed_count, len(configs), result)

        return results

    def run_from_yaml(
        self,
        sweep_config_path: str | Path,
        progress_callback: Callable[[int, int, RunResult | None], None] | None = None,
    ) -> SweepResult:
        """
        Run a sweep from a YAML configuration file.

        Args:
            sweep_config_path: Path to sweep config YAML
            progress_callback: Progress callback

        Returns:
            SweepResult
        """
        sweep_config = SweepConfig.from_yaml(str(sweep_config_path))
        return self.run_sweep(sweep_config, progress_callback)

    def quick_sweep(
        self,
        parameters: list[SweepParameter],
        name: str = "quick_sweep",
        dataset: Literal["bioasq", "pubmedqa"] = "bioasq",
        max_questions: int = 10,
        seed: int = 42,
    ) -> SweepResult:
        """
        Run a quick sweep for testing.

        Args:
            parameters: Parameters to sweep
            name: Sweep name
            dataset: Dataset to evaluate on
            max_questions: Maximum questions per run
            seed: Random seed

        Returns:
            SweepResult
        """
        sweep_config = SweepConfig(
            name=name,
            parameters=parameters,
            dataset=dataset,
            max_questions=max_questions,
            seed=seed,
            save_artifacts=False,
        )
        return self.run_sweep(sweep_config)


def create_parameter(
    path: str,
    values: list[Any] | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
    step: float | None = None,
    num: int | None = None,
) -> SweepParameter:
    """
    Helper function to create a SweepParameter.

    Args:
        path: Dot-separated config path
        values: Explicit list of values
        min_val: Minimum value for range
        max_val: Maximum value for range
        step: Step size for range
        num: Number of values for range

    Returns:
        SweepParameter
    """
    from biorag.schemas.experiments import ParameterRange

    if values is not None:
        return SweepParameter(
            path=path,
            range=ParameterRange(type="grid", values=values),
        )
    elif min_val is not None and max_val is not None:
        return SweepParameter(
            path=path,
            range=ParameterRange(
                type="range",
                min=min_val,
                max=max_val,
                step=step,
                num=num,
            ),
        )
    else:
        raise ValueError("Must provide either 'values' or 'min_val' and 'max_val'")

