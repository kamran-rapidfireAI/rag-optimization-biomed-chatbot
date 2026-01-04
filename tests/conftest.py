"""Pytest configuration and shared fixtures for BioRAG Bench tests."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from biorag.schemas.config import BioRAGConfig


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def default_config() -> BioRAGConfig:
    """Return a default BioRAG configuration."""
    return BioRAGConfig()


@pytest.fixture
def test_config(temp_dir: Path) -> BioRAGConfig:
    """Return a test configuration with temporary paths."""
    return BioRAGConfig(
        paths={
            "data_dir": temp_dir / "data",
            "runs_dir": temp_dir / "runs",
            "cache_dir": temp_dir / "cache",
        },
        logging={
            "level": "DEBUG",
            "json_format": False,
        },
    )


@pytest.fixture
def sample_config_yaml(temp_dir: Path) -> Path:
    """Create a sample YAML config file."""
    config_path = temp_dir / "test_config.yaml"
    config_content = """
llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 350

embeddings:
  provider: openai
  model: text-embedding-3-large

chunking:
  type: recursive
  chunk_size: 350
  chunk_overlap: 40

retrieval:
  mode: mmr
  k: 10
  fetch_k: 50
  lambda_mult: 0.5

rerank:
  enabled: true
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_n: 50
  final_k: 8

logging:
  level: INFO
  json_format: false
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture(autouse=True)
def reset_env() -> Generator[None, None, None]:
    """Reset environment variables after each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_openai_key() -> Generator[None, None, None]:
    """Set a mock OpenAI API key for tests."""
    os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing-only"
    yield
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]


# Markers for test categorization
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
