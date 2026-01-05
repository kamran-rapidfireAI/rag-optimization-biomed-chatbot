"""Artifact management for experiment runs.

Handles saving, loading, and organizing experiment artifacts including:
- run.json with full reproducibility info
- predictions.jsonl with per-question outputs
- metrics.json with aggregated metrics
- leaderboard.csv with ranked configurations
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biorag.schemas.config import BioRAGConfig
from biorag.utils.logging import get_logger

if TYPE_CHECKING:
    from biorag.experiments.runner import RunResult

logger = get_logger(__name__)


def get_git_commit() -> str | None:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return None


def get_git_branch() -> str | None:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_git_dirty() -> bool:
    """Check if git working directory is dirty."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


@dataclass
class ReproducibilityInfo:
    """Information needed to reproduce an experiment."""

    git_commit: str | None
    git_branch: str | None
    git_dirty: bool
    timestamp: str
    config: dict[str, Any]
    model_versions: dict[str, str]
    dataset: str
    dataset_version: str | None
    random_seed: int
    total_tokens: int
    estimated_cost_usd: float
    cache_hit_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "timestamp": self.timestamp,
            "config": self.config,
            "model_versions": self.model_versions,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "random_seed": self.random_seed,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cache_hit_rate": self.cache_hit_rate,
        }

    @classmethod
    def from_run_result(cls, result: "RunResult", dataset: str = "bioasq") -> "ReproducibilityInfo":
        """Create from a RunResult."""
        metrics = result.eval_result.metrics
        total_tokens = 0
        estimated_cost = 0.0
        cache_hit_rate = 0.0

        if metrics:
            total_tokens = metrics.total_input_tokens + metrics.total_output_tokens
            estimated_cost = metrics.estimated_cost_usd
            cache_hit_rate = metrics.cache_hit_rate

        return cls(
            git_commit=get_git_commit(),
            git_branch=get_git_branch(),
            git_dirty=get_git_dirty(),
            timestamp=result.timestamp.isoformat(),
            config=result.config.model_dump(mode="json"),
            model_versions=result.eval_result.model_versions,
            dataset=dataset,
            dataset_version=result.eval_result.dataset_version,
            random_seed=result.eval_result.random_seed,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
            cache_hit_rate=cache_hit_rate,
        )


class ArtifactManager:
    """
    Manages experiment artifacts for runs and sweeps.

    Features:
    - Save/load run.json with full reproducibility info
    - Generate leaderboard.csv ranking configurations
    - Organize artifacts by run ID
    """

    def __init__(self, output_dir: str | Path) -> None:
        """
        Initialize artifact manager.

        Args:
            output_dir: Base directory for artifacts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ArtifactManager initialized: output_dir={self.output_dir}")

    def save_run(
        self,
        result: "RunResult",
        dataset: str = "bioasq",
    ) -> Path:
        """
        Save all artifacts for a run.

        Args:
            result: RunResult to save
            dataset: Dataset name for reproducibility info

        Returns:
            Path to the run directory
        """
        run_dir = self.output_dir / result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build reproducibility info
        repro_info = ReproducibilityInfo.from_run_result(result, dataset)

        # Save run.json with full info
        run_data = {
            "run_id": result.run_id,
            "status": result.status,
            "error": result.error,
            "primary_metric": result.primary_metric,
            "reproducibility": repro_info.to_dict(),
            "latency": result.latency.to_dict(),
            "metrics": result.eval_result.metrics.model_dump(mode="json") if result.eval_result.metrics else None,
        }

        with open(run_dir / "run.json", "w") as f:
            json.dump(run_data, f, indent=2, default=str)

        # Save config.yaml for easy inspection
        config_path = run_dir / "config.yaml"
        result.config.to_yaml(config_path)

        # Save predictions if available
        if result.eval_result.predictions:
            with open(run_dir / "predictions.jsonl", "w") as f:
                for pred in result.eval_result.predictions:
                    f.write(json.dumps(pred.model_dump(mode="json"), default=str) + "\n")

        # Save metrics summary
        if result.eval_result.metrics:
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(result.eval_result.metrics.model_dump(mode="json"), f, indent=2, default=str)

        logger.info(f"Saved run artifacts to {run_dir}")
        return run_dir

    def load_run(self, run_id: str) -> dict[str, Any]:
        """
        Load a run's artifacts.

        Args:
            run_id: Run identifier

        Returns:
            Dictionary with run data
        """
        run_dir = self.output_dir / run_id

        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")

        with open(run_dir / "run.json") as f:
            return json.load(f)

    def list_runs(self) -> list[str]:
        """List all run IDs in the output directory."""
        runs = []
        for path in self.output_dir.iterdir():
            if path.is_dir() and (path / "run.json").exists():
                runs.append(path.name)
        return sorted(runs)

    def generate_leaderboard(
        self,
        results: list["RunResult"],
        output_path: str | Path | None = None,
        primary_metric: str = "primary_metric",
        include_config_params: bool = True,
    ) -> Path:
        """
        Generate a leaderboard CSV from run results.

        Args:
            results: List of RunResult objects
            output_path: Path to save leaderboard (default: output_dir/leaderboard.csv)
            primary_metric: Column to sort by
            include_config_params: Whether to include key config parameters

        Returns:
            Path to the leaderboard file
        """
        output_path = Path(output_path) if output_path else self.output_dir / "leaderboard.csv"

        # Build leaderboard rows
        rows: list[dict[str, Any]] = []

        for result in results:
            row: dict[str, Any] = {
                "rank": 0,  # Will be set after sorting
                "run_id": result.run_id,
                "status": result.status,
                "primary_metric": result.primary_metric,
                "total_time_ms": result.latency.total_ms,
                "timestamp": result.timestamp.isoformat(),
            }

            # Add metrics
            if result.eval_result.metrics:
                metrics = result.eval_result.metrics
                row["num_questions"] = metrics.num_questions
                row["num_abstained"] = metrics.num_abstained
                row["estimated_cost_usd"] = metrics.estimated_cost_usd
                row["cache_hit_rate"] = metrics.cache_hit_rate

                # Add all answer metrics
                for name, metric in metrics.answer_metrics.items():
                    row[f"metric_{name}"] = metric.value

            # Add key config parameters
            if include_config_params:
                config = result.config
                row["llm_model"] = config.llm.model
                row["embeddings_model"] = config.embeddings.model
                row["chunking_type"] = config.chunking.type
                row["chunk_size"] = config.chunking.chunk_size
                row["chunk_overlap"] = config.chunking.chunk_overlap
                row["retrieval_mode"] = config.retrieval.mode
                row["retrieval_k"] = config.retrieval.k
                row["rerank_enabled"] = config.rerank.enabled
                row["rerank_model"] = config.rerank.model if config.rerank.enabled else "none"
                row["rerank_final_k"] = config.rerank.final_k

            rows.append(row)

        # Sort by primary metric (descending) and assign ranks
        rows.sort(key=lambda x: x.get(primary_metric, 0), reverse=True)
        for i, row in enumerate(rows):
            row["rank"] = i + 1

        # Write CSV (create file even if empty)
        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            # Create empty file with header only
            with open(output_path, "w", newline="") as f:
                f.write("rank,run_id,status,primary_metric\n")

        logger.info(f"Leaderboard saved to {output_path} ({len(rows)} runs)")
        return output_path

    def load_leaderboard(self, path: str | Path | None = None) -> list[dict[str, Any]]:
        """
        Load leaderboard from CSV.

        Args:
            path: Path to leaderboard file (default: output_dir/leaderboard.csv)

        Returns:
            List of leaderboard rows as dictionaries
        """
        path = Path(path) if path else self.output_dir / "leaderboard.csv"

        if not path.exists():
            return []

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def get_best_config(
        self,
        results: list["RunResult"] | None = None,
        leaderboard_path: str | Path | None = None,
    ) -> dict[str, Any] | None:
        """
        Get the best performing configuration.

        Args:
            results: List of RunResult objects (if provided, skips leaderboard)
            leaderboard_path: Path to leaderboard CSV

        Returns:
            Best config as dictionary, or None if no runs found
        """
        if results:
            # Get from results
            if not results:
                return None
            best = max(results, key=lambda r: r.primary_metric)
            return best.config.model_dump(mode="json")

        # Get from leaderboard
        leaderboard = self.load_leaderboard(leaderboard_path)
        if not leaderboard:
            return None

        # Load the best run's config
        best_run_id = leaderboard[0]["run_id"]
        run_data = self.load_run(best_run_id)
        return run_data.get("reproducibility", {}).get("config")


def save_sweep_summary(
    results: list["RunResult"],
    output_path: str | Path,
    sweep_config: dict[str, Any],
) -> Path:
    """
    Save a summary of a sweep run.

    Args:
        results: List of RunResult objects
        output_path: Path to save summary
        sweep_config: Original sweep configuration

    Returns:
        Path to the summary file
    """
    output_path = Path(output_path)

    # Compute summary statistics
    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]

    if completed:
        best = max(completed, key=lambda r: r.primary_metric)
        worst = min(completed, key=lambda r: r.primary_metric)
        avg_metric = sum(r.primary_metric for r in completed) / len(completed)
    else:
        best = worst = None
        avg_metric = 0.0

    summary = {
        "sweep_timestamp": datetime.utcnow().isoformat(),
        "git_commit": get_git_commit(),
        "git_branch": get_git_branch(),
        "sweep_config": sweep_config,
        "total_runs": len(results),
        "completed_runs": len(completed),
        "failed_runs": len(failed),
        "best_run": {
            "run_id": best.run_id if best else None,
            "primary_metric": best.primary_metric if best else None,
            "config": best.config.model_dump(mode="json") if best else None,
        },
        "worst_run": {
            "run_id": worst.run_id if worst else None,
            "primary_metric": worst.primary_metric if worst else None,
        },
        "average_metric": avg_metric,
        "total_cost_usd": sum(
            r.eval_result.metrics.estimated_cost_usd
            for r in completed
            if r.eval_result.metrics
        ),
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Sweep summary saved to {output_path}")
    return output_path

