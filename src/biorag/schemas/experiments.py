"""Schemas for experiment sweeps and configurations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ParameterRange(BaseModel):
    """Definition of a parameter range for sweeping."""

    type: Literal["grid", "choice", "range"] = "grid"
    values: list[Any] | None = Field(
        default=None,
        description="Explicit list of values for grid/choice",
    )
    min: float | None = Field(default=None, description="Minimum value for range")
    max: float | None = Field(default=None, description="Maximum value for range")
    step: float | None = Field(default=None, description="Step size for range")
    num: int | None = Field(default=None, description="Number of values for range")

    def get_values(self) -> list[Any]:
        """Get all values in this parameter range."""
        if self.values is not None:
            return list(self.values)

        if self.type == "range" and self.min is not None and self.max is not None:
            if self.step is not None:
                # Use step
                values = []
                current = self.min
                while current <= self.max:
                    values.append(current)
                    current += self.step
                return values
            elif self.num is not None:
                # Use num
                if self.num == 1:
                    return [self.min]
                step = (self.max - self.min) / (self.num - 1)
                return [self.min + i * step for i in range(self.num)]

        return []


class SweepParameter(BaseModel):
    """A parameter to sweep over."""

    path: str = Field(
        ...,
        description="Dot-separated path to config parameter (e.g., 'chunking.chunk_size')",
    )
    range: ParameterRange = Field(..., description="Parameter range to sweep")

    def get_values(self) -> list[Any]:
        """Get all values for this parameter."""
        return self.range.get_values()


class SweepConfig(BaseModel):
    """Configuration for a parameter sweep."""

    name: str = Field(..., description="Name of the sweep")
    description: str = Field(default="", description="Description of the sweep")

    # Base configuration path (optional)
    base_config: str | None = Field(
        default=None,
        description="Path to base configuration file",
    )

    # Parameters to sweep
    parameters: list[SweepParameter] = Field(
        default_factory=list,
        description="Parameters to sweep over",
    )

    # Evaluation settings
    dataset: Literal["bioasq", "pubmedqa"] = Field(
        default="bioasq",
        description="Dataset to evaluate on",
    )
    split: str = Field(default="train", description="Dataset split")
    max_questions: int | None = Field(
        default=None,
        description="Maximum questions per run",
    )
    seed: int = Field(default=42, description="Random seed")

    # Execution settings
    parallel: bool = Field(
        default=False,
        description="Whether to run configs in parallel (requires RapidFire AI)",
    )
    max_parallel: int = Field(
        default=4,
        description="Maximum parallel runs",
    )
    save_artifacts: bool = Field(
        default=True,
        description="Whether to save run artifacts",
    )

    # Output settings
    output_dir: str | None = Field(
        default=None,
        description="Output directory for sweep results",
    )

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: str) -> "SweepConfig":
        """Load sweep config from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str) -> None:
        """Save sweep config to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)

    def get_num_configs(self) -> int:
        """Get total number of configurations in the grid."""
        if not self.parameters:
            return 1
        total = 1
        for param in self.parameters:
            total *= len(param.get_values())
        return total


class SweepResult(BaseModel):
    """Result of a parameter sweep."""

    sweep_name: str
    sweep_config: dict[str, Any]
    total_runs: int
    completed_runs: int
    failed_runs: int
    best_run_id: str | None
    best_metric: float | None
    best_config: dict[str, Any] | None
    average_metric: float
    total_cost_usd: float
    leaderboard_path: str | None

    model_config = {"extra": "ignore"}





