"""Configuration schemas for BioRAG Bench pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    """Configuration for the LLM provider."""

    provider: Literal["openai", "azure", "local"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=350, ge=1, le=4096)
    timeout: float = Field(default=30.0, ge=1.0, description="Timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10)


class EmbeddingsConfig(BaseModel):
    """Configuration for embeddings provider."""

    provider: Literal["openai", "local", "huggingface"] = "openai"
    model: str = "text-embedding-3-large"
    batch_size: int = Field(default=100, ge=1, le=2048)
    cache_enabled: bool = True


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""

    type: Literal["recursive", "token", "sentence"] = "recursive"
    chunk_size: int = Field(default=350, ge=50, le=2000)
    chunk_overlap: int = Field(default=40, ge=0, le=500)
    separators: list[str] | None = None


class RetrievalConfig(BaseModel):
    """Configuration for retrieval."""

    mode: Literal["similarity", "mmr", "similarity_score_threshold"] = "mmr"
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    fetch_k: int = Field(default=50, ge=1, le=500, description="Number to fetch for MMR")
    lambda_mult: float = Field(
        default=0.5, ge=0.0, le=1.0, description="MMR diversity parameter"
    )
    score_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Score threshold for filtering"
    )


class RerankConfig(BaseModel):
    """Configuration for reranking."""

    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = Field(default=50, ge=1, le=200, description="Number to rerank")
    final_k: int = Field(default=8, ge=1, le=50, description="Final number after reranking")
    batch_size: int = Field(default=32, ge=1, le=256)


class PromptConfig(BaseModel):
    """Configuration for prompt management."""

    template: str = "prompts/cite_and_abstain_v2.txt"
    citation_policy: Literal["strict", "claim-level"] = "claim-level"


class AbstentionConfig(BaseModel):
    """Configuration for abstention logic."""

    min_evidence_score: float = Field(default=0.3, ge=0.0, le=1.0)
    min_evidence_chunks: int = Field(default=1, ge=0, le=10)
    enable_self_check: bool = True


class CostConfig(BaseModel):
    """Configuration for cost controls."""

    max_questions: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_usd: float | None = Field(default=None, ge=0.0)
    on_budget_exceeded: Literal["fail-fast", "skip"] = "fail-fast"


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = False
    include_timestamps: bool = True
    log_file: str | None = None


class PathsConfig(BaseModel):
    """Configuration for paths."""

    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")
    cache_dir: Path = Path("data/cache")
    configs_dir: Path = Path("configs")

    def ensure_dirs(self) -> None:
        """Create all configured directories if they don't exist."""
        for path in [self.data_dir, self.runs_dir, self.cache_dir]:
            path.mkdir(parents=True, exist_ok=True)


class BioRAGConfig(BaseModel):
    """Main configuration for BioRAG Bench pipeline."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: str | Path) -> BioRAGConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        # Convert to dict with mode='json' to serialize Path objects as strings
        data = self.model_dump(mode="json")
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def merge_with(self, overrides: dict) -> BioRAGConfig:
        """Create a new config with overrides applied."""
        current = self.model_dump()
        self._deep_merge(current, overrides)
        return BioRAGConfig.model_validate(current)

    @staticmethod
    def _deep_merge(base: dict, overrides: dict) -> None:
        """Recursively merge overrides into base dict."""
        for key, value in overrides.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                BioRAGConfig._deep_merge(base[key], value)
            else:
                base[key] = value


class EnvSettings(BaseSettings):
    """Environment-based settings that override config file values."""

    openai_api_key: str | None = None
    biorag_log_level: str = "INFO"
    biorag_json_logs: bool = False
    biorag_data_dir: str = "data"
    biorag_runs_dir: str = "runs"
    biorag_cache_dir: str = "data/cache"
    hf_token: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_config(config_path: str | Path | None = None) -> BioRAGConfig:
    """
    Load configuration from file with environment variable overrides.

    Args:
        config_path: Path to YAML config file. If None, uses defaults.

    Returns:
        BioRAGConfig instance with all settings applied.
    """
    # Load base config from file or use defaults
    config = (
        BioRAGConfig.from_yaml(config_path) if config_path is not None else BioRAGConfig()
    )

    # Load environment settings
    env = EnvSettings()

    # Apply environment overrides
    overrides: dict = {
        "logging": {
            "level": env.biorag_log_level,
            "json_format": env.biorag_json_logs,
        },
        "paths": {
            "data_dir": env.biorag_data_dir,
            "runs_dir": env.biorag_runs_dir,
            "cache_dir": env.biorag_cache_dir,
        },
    }

    return config.merge_with(overrides)

