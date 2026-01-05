"""Experiment runner and sweep management for BioRAG Bench.

This module integrates with RapidFire AI for hyperparallelized experimentation
when available, with fallback to sequential execution.
"""

from biorag.experiments.artifacts import (
    ArtifactManager,
    ReproducibilityInfo,
    get_git_branch,
    get_git_commit,
    get_git_dirty,
    save_sweep_summary,
)
from biorag.experiments.runner import (
    ExperimentRunner,
    RunLatency,
    RunResult,
)
from biorag.experiments.sweep import (
    RAPIDFIRE_AVAILABLE,
    SweepRunner,
    create_parameter,
    generate_grid,
)

# RapidFire AI integration (optional)
try:
    from biorag.experiments.rapidfire_adapter import (
        BioRAGConfigAdapter,
        BioRAGDatasetAdapter,
        RapidFireSweepRunner,
        check_rapidfire_available,
    )
except ImportError:
    BioRAGConfigAdapter = None  # type: ignore
    BioRAGDatasetAdapter = None  # type: ignore
    RapidFireSweepRunner = None  # type: ignore
    check_rapidfire_available = lambda: False  # type: ignore

__all__ = [
    # Runner
    "ExperimentRunner",
    "RunResult",
    "RunLatency",
    # Artifacts
    "ArtifactManager",
    "ReproducibilityInfo",
    "get_git_commit",
    "get_git_branch",
    "get_git_dirty",
    "save_sweep_summary",
    # Sweep
    "SweepRunner",
    "generate_grid",
    "create_parameter",
    "RAPIDFIRE_AVAILABLE",
    # RapidFire AI (when available)
    "RapidFireSweepRunner",
    "BioRAGConfigAdapter",
    "BioRAGDatasetAdapter",
    "check_rapidfire_available",
]




