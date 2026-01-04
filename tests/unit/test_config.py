"""Unit tests for configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest

from biorag.schemas.config import (
    BioRAGConfig,
    ChunkingConfig,
    EmbeddingsConfig,
    LLMConfig,
    LoggingConfig,
    PathsConfig,
    RerankConfig,
    RetrievalConfig,
    load_config,
)


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_values(self) -> None:
        """Test default LLM config values."""
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_tokens == 350
        assert config.timeout == 30.0
        assert config.max_retries == 3

    def test_custom_values(self) -> None:
        """Test custom LLM config values."""
        config = LLMConfig(
            provider="azure",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
        )
        assert config.provider == "azure"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000

    def test_temperature_validation(self) -> None:
        """Test temperature value bounds."""
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            LLMConfig(temperature=2.1)


class TestEmbeddingsConfig:
    """Tests for EmbeddingsConfig."""

    def test_default_values(self) -> None:
        """Test default embeddings config."""
        config = EmbeddingsConfig()
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-large"
        assert config.batch_size == 100
        assert config.cache_enabled is True


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_values(self) -> None:
        """Test default chunking config."""
        config = ChunkingConfig()
        assert config.type == "recursive"
        assert config.chunk_size == 350
        assert config.chunk_overlap == 40

    def test_chunk_size_bounds(self) -> None:
        """Test chunk size validation."""
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=10)  # Too small
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=5000)  # Too large


class TestRetrievalConfig:
    """Tests for RetrievalConfig."""

    def test_default_values(self) -> None:
        """Test default retrieval config."""
        config = RetrievalConfig()
        assert config.mode == "mmr"
        assert config.k == 10
        assert config.fetch_k == 50
        assert config.lambda_mult == 0.5

    def test_lambda_mult_bounds(self) -> None:
        """Test lambda_mult validation."""
        with pytest.raises(ValueError):
            RetrievalConfig(lambda_mult=-0.1)
        with pytest.raises(ValueError):
            RetrievalConfig(lambda_mult=1.5)


class TestRerankConfig:
    """Tests for RerankConfig."""

    def test_default_values(self) -> None:
        """Test default rerank config."""
        config = RerankConfig()
        assert config.enabled is True
        assert config.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert config.top_n == 50
        assert config.final_k == 8


class TestBioRAGConfig:
    """Tests for main BioRAGConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = BioRAGConfig()
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.embeddings, EmbeddingsConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.retrieval, RetrievalConfig)
        assert isinstance(config.rerank, RerankConfig)

    def test_from_yaml(self, sample_config_yaml: Path) -> None:
        """Test loading config from YAML file."""
        config = BioRAGConfig.from_yaml(sample_config_yaml)
        assert config.llm.model == "gpt-4o-mini"
        assert config.embeddings.model == "text-embedding-3-large"
        assert config.chunking.chunk_size == 350
        assert config.retrieval.mode == "mmr"

    def test_to_yaml(self, temp_dir: Path) -> None:
        """Test saving config to YAML file."""
        config = BioRAGConfig()
        output_path = temp_dir / "output_config.yaml"
        config.to_yaml(output_path)
        assert output_path.exists()

        # Reload and verify
        loaded = BioRAGConfig.from_yaml(output_path)
        assert loaded.llm.model == config.llm.model
        assert loaded.chunking.chunk_size == config.chunking.chunk_size

    def test_merge_with(self) -> None:
        """Test merging config with overrides."""
        config = BioRAGConfig()
        overrides = {
            "llm": {"model": "gpt-4"},
            "chunking": {"chunk_size": 500},
        }
        merged = config.merge_with(overrides)

        assert merged.llm.model == "gpt-4"
        assert merged.chunking.chunk_size == 500
        # Original should be unchanged
        assert config.llm.model == "gpt-4o-mini"
        assert config.chunking.chunk_size == 350

    def test_nested_merge(self) -> None:
        """Test deep merging of nested config."""
        config = BioRAGConfig()
        overrides = {
            "retrieval": {
                "k": 20,
                # lambda_mult should remain default
            }
        }
        merged = config.merge_with(overrides)

        assert merged.retrieval.k == 20
        assert merged.retrieval.lambda_mult == 0.5  # Default preserved

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValueError):
            BioRAGConfig(unknown_field="value")


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_default_values(self) -> None:
        """Test default logging config."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.json_format is False
        assert config.include_timestamps is True
        assert config.log_file is None

    def test_valid_levels(self) -> None:
        """Test valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            config = LoggingConfig(level=level)
            assert config.level == level


class TestPathsConfig:
    """Tests for PathsConfig."""

    def test_default_paths(self) -> None:
        """Test default paths."""
        config = PathsConfig()
        assert config.data_dir == Path("data")
        assert config.runs_dir == Path("runs")
        assert config.cache_dir == Path("data/cache")

    def test_ensure_dirs(self, temp_dir: Path) -> None:
        """Test directory creation."""
        config = PathsConfig(
            data_dir=temp_dir / "data",
            runs_dir=temp_dir / "runs",
            cache_dir=temp_dir / "cache",
        )
        config.ensure_dirs()

        assert (temp_dir / "data").exists()
        assert (temp_dir / "runs").exists()
        assert (temp_dir / "cache").exists()


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default(self) -> None:
        """Test loading default config."""
        config = load_config()
        assert isinstance(config, BioRAGConfig)

    def test_load_from_file(self, sample_config_yaml: Path) -> None:
        """Test loading config from file."""
        config = load_config(sample_config_yaml)
        assert config.llm.model == "gpt-4o-mini"

