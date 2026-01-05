"""Single run executor for experiment sweeps.

Executes one configuration against a golden suite and records all artifacts.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from biorag.eval.harness import EvaluationHarness, EvalProgress
from biorag.schemas.config import BioRAGConfig, load_config
from biorag.schemas.evaluation import EvalResult
from biorag.utils.caching import LLMCache
from biorag.utils.cost import CostTracker
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RunLatency:
    """Latency breakdown for a run."""

    pipeline_setup_ms: float = 0.0
    index_load_ms: float = 0.0
    evaluation_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "pipeline_setup_ms": self.pipeline_setup_ms,
            "index_load_ms": self.index_load_ms,
            "evaluation_ms": self.evaluation_ms,
            "total_ms": self.total_ms,
        }


@dataclass
class RunResult:
    """Result of a single experiment run."""

    run_id: str
    config: BioRAGConfig
    eval_result: EvalResult
    latency: RunLatency
    status: Literal["completed", "failed", "partial"] = "completed"
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def primary_metric(self) -> float:
        """Get primary metric for ranking (defaults to accuracy or F1)."""
        if self.eval_result.metrics:
            # Try common primary metrics
            for metric_name in ["accuracy", "token_f1", "exact_match", "macro_f1"]:
                if metric_name in self.eval_result.metrics.answer_metrics:
                    return self.eval_result.metrics.answer_metrics[metric_name].value
            # Fallback: return first available metric
            if self.eval_result.metrics.answer_metrics:
                return list(self.eval_result.metrics.answer_metrics.values())[0].value
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "config": self.config.model_dump(mode="json"),
            "eval_result": self.eval_result.model_dump(mode="json"),
            "latency": self.latency.to_dict(),
            "status": self.status,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "primary_metric": self.primary_metric,
        }


class ExperimentRunner:
    """
    Executes a single experiment configuration against a golden suite.

    Features:
    - Full artifact tracking (config, predictions, metrics)
    - Latency breakdown per stage
    - Cost tracking and budget enforcement
    - LLM output caching across runs
    """

    def __init__(
        self,
        base_config: BioRAGConfig | None = None,
        base_config_path: str | Path | None = None,
        cache: LLMCache | None = None,
        output_dir: str | Path | None = None,
        index_path: str | Path | None = None,
    ) -> None:
        """
        Initialize experiment runner.

        Args:
            base_config: Base configuration to override
            base_config_path: Path to base config file
            cache: Shared LLM cache across runs
            output_dir: Directory for saving run artifacts
            index_path: Path to pre-built FAISS index
        """
        # Load base config
        if base_config is not None:
            self.base_config = base_config
        elif base_config_path is not None:
            self.base_config = load_config(base_config_path)
        else:
            self.base_config = load_config()

        # Shared cache across runs for efficiency
        self.cache = cache or LLMCache(
            cache_dir=self.base_config.paths.cache_dir / "llm_cache"
        )

        # Output directory
        self.output_dir = Path(output_dir) if output_dir else self.base_config.paths.runs_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Index path (if using pre-built index)
        self.index_path = Path(index_path) if index_path else None

        logger.info(f"ExperimentRunner initialized: output_dir={self.output_dir}")

    def run(
        self,
        config_overrides: dict[str, Any] | None = None,
        dataset: Literal["bioasq", "pubmedqa"] = "bioasq",
        split: str = "train",
        max_questions: int | None = None,
        run_id: str | None = None,
        seed: int = 42,
        progress_callback: Callable[[EvalProgress], None] | None = None,
        save_artifacts: bool = True,
    ) -> RunResult:
        """
        Execute a single experiment run.

        Args:
            config_overrides: Dictionary of config overrides to apply
            dataset: Dataset to evaluate on
            split: Dataset split
            max_questions: Maximum questions to evaluate
            run_id: Optional run identifier
            seed: Random seed for sampling
            progress_callback: Callback for progress updates
            save_artifacts: Whether to save artifacts to disk

        Returns:
            RunResult with eval results and metadata
        """
        # Generate run ID
        run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        start_time = time.time()
        latency = RunLatency()

        logger.info(f"Starting run: {run_id}")

        try:
            # Apply config overrides
            config = self._apply_overrides(config_overrides or {})
            logger.debug(f"Config overrides: {config_overrides}")

            # Create cost tracker for this run
            cost_tracker = CostTracker(
                max_usd=config.cost.max_usd,
                max_total_tokens=config.cost.max_total_tokens,
                on_budget_exceeded=config.cost.on_budget_exceeded,
            )

            # Create evaluation harness
            setup_start = time.time()
            harness = EvaluationHarness(
                config=config,
                cache=self.cache,
                cost_tracker=cost_tracker,
                output_dir=self.output_dir / run_id,
            )
            latency.pipeline_setup_ms = (time.time() - setup_start) * 1000

            # Load index if available
            if self.index_path:
                index_start = time.time()
                harness.pipeline.load_index(self.index_path)
                latency.index_load_ms = (time.time() - index_start) * 1000

            # Run evaluation
            eval_start = time.time()
            if dataset == "bioasq":
                questions = harness.load_golden_suite(
                    dataset="bioasq",
                    split=split,
                    max_questions=max_questions,
                    seed=seed,
                )
                eval_result = harness.evaluate_bioasq(
                    questions,  # type: ignore
                    run_id=run_id,
                    progress_callback=progress_callback,
                    save_results=save_artifacts,
                )
            else:
                questions = harness.load_golden_suite(
                    dataset="pubmedqa",
                    split=split,
                    max_questions=max_questions,
                    seed=seed,
                )
                eval_result = harness.evaluate_pubmedqa(
                    questions,  # type: ignore
                    run_id=run_id,
                    progress_callback=progress_callback,
                    save_results=save_artifacts,
                )

            latency.evaluation_ms = (time.time() - eval_start) * 1000
            latency.total_ms = (time.time() - start_time) * 1000

            # Create result
            result = RunResult(
                run_id=run_id,
                config=config,
                eval_result=eval_result,
                latency=latency,
                status="completed",
            )

            logger.info(
                f"Run {run_id} completed: "
                f"primary_metric={result.primary_metric:.4f}, "
                f"total_time={latency.total_ms:.1f}ms"
            )

            return result

        except Exception as e:
            latency.total_ms = (time.time() - start_time) * 1000
            logger.error(f"Run {run_id} failed: {e}")

            # Create failed result
            return RunResult(
                run_id=run_id,
                config=self._apply_overrides(config_overrides or {}),
                eval_result=EvalResult(run_id=run_id),
                latency=latency,
                status="failed",
                error=str(e),
            )

    def _apply_overrides(self, overrides: dict[str, Any]) -> BioRAGConfig:
        """Apply config overrides to base config."""
        return self.base_config.merge_with(overrides)

    def run_batch(
        self,
        configs: list[dict[str, Any]],
        dataset: Literal["bioasq", "pubmedqa"] = "bioasq",
        split: str = "train",
        max_questions: int | None = None,
        seed: int = 42,
        save_artifacts: bool = True,
    ) -> list[RunResult]:
        """
        Run multiple configurations sequentially.

        For parallel execution, use the SweepRunner class.

        Args:
            configs: List of config override dictionaries
            dataset: Dataset to evaluate on
            split: Dataset split
            max_questions: Max questions per run
            seed: Random seed
            save_artifacts: Whether to save artifacts

        Returns:
            List of RunResult objects
        """
        results: list[RunResult] = []

        for i, config_overrides in enumerate(configs):
            logger.info(f"Running configuration {i + 1}/{len(configs)}")
            result = self.run(
                config_overrides=config_overrides,
                dataset=dataset,
                split=split,
                max_questions=max_questions,
                seed=seed,
                save_artifacts=save_artifacts,
            )
            results.append(result)

        return results





