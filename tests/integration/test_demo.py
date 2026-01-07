"""Integration tests for the Gradio demo (Phase 8)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add demo directory to path
DEMO_DIR = Path(__file__).parent.parent.parent / "demo"
sys.path.insert(0, str(DEMO_DIR))


class TestBioRAGDemo:
    """Tests for BioRAGDemo class."""

    def test_demo_imports(self) -> None:
        """Test that demo module can be imported."""
        from demo.app import (
            BioRAGDemo,
            create_demo,
            format_chunk,
            format_citation,
            format_latency,
            format_config_summary,
            BASELINE_CONFIG,
            OPTIMIZED_CONFIG,
            DISCLAIMER_HTML,
        )
        
        assert BioRAGDemo is not None
        assert create_demo is not None
        assert BASELINE_CONFIG is not None
        assert OPTIMIZED_CONFIG is not None

    def test_baseline_config_structure(self) -> None:
        """Test baseline config has expected structure."""
        from demo.app import BASELINE_CONFIG
        
        assert "retrieval" in BASELINE_CONFIG
        assert "rerank" in BASELINE_CONFIG
        assert BASELINE_CONFIG["retrieval"]["mode"] == "similarity"
        assert BASELINE_CONFIG["rerank"]["enabled"] is False

    def test_optimized_config_structure(self) -> None:
        """Test optimized config has expected structure."""
        from demo.app import OPTIMIZED_CONFIG
        
        assert "retrieval" in OPTIMIZED_CONFIG
        assert "rerank" in OPTIMIZED_CONFIG
        assert OPTIMIZED_CONFIG["retrieval"]["mode"] == "mmr"
        assert OPTIMIZED_CONFIG["rerank"]["enabled"] is True
        assert "model" in OPTIMIZED_CONFIG["rerank"]

    def test_format_citation_with_quote(self) -> None:
        """Test citation formatting with quote."""
        from demo.app import format_citation
        
        citation = {
            "pmid": "12345678",
            "quote": "This is a sample quote from the paper that should be truncated...",
        }
        
        result = format_citation(citation, 1)
        
        assert "**[1]**" in result
        assert "PMID:12345678" in result
        assert "quote" in result.lower() or "sample" in result.lower()

    def test_format_citation_without_quote(self) -> None:
        """Test citation formatting without quote."""
        from demo.app import format_citation
        
        citation = {"pmid": "12345678"}
        
        result = format_citation(citation, 2)
        
        assert "**[2]**" in result
        assert "PMID:12345678" in result

    def test_format_chunk_with_rerank(self) -> None:
        """Test chunk formatting with rerank scores."""
        from demo.app import format_chunk
        
        chunk = {
            "pmid": "12345678",
            "text": "Sample text for the chunk that contains medical information.",
            "score": 0.85,
            "rank": 3,
            "rerank_score": 0.95,
            "rerank_rank": 1,
        }
        
        result = format_chunk(chunk, 0, show_rerank=True)
        
        assert "**#1**" in result  # rerank_rank
        assert "PMID:12345678" in result
        assert "0.95" in result  # rerank_score
        assert "Sample text" in result

    def test_format_chunk_without_rerank(self) -> None:
        """Test chunk formatting without rerank scores."""
        from demo.app import format_chunk
        
        chunk = {
            "pmid": "12345678",
            "text": "Sample text for the chunk.",
            "score": 0.85,
            "rank": 3,
        }
        
        result = format_chunk(chunk, 0, show_rerank=False)
        
        assert "**#3**" in result  # original rank
        assert "0.85" in result  # original score

    def test_format_latency_with_rerank(self) -> None:
        """Test latency formatting with reranking."""
        from demo.app import format_latency
        
        latency = {
            "retrieve_ms": 50.0,
            "rerank_ms": 20.0,
            "generate_ms": 4000.0,
            "total_ms": 4070.0,
        }
        
        result = format_latency(latency, show_rerank=True)
        
        assert "Retrieval" in result
        assert "50.0 ms" in result
        assert "Reranking" in result
        assert "20.0 ms" in result
        assert "Generation" in result
        assert "4000.0 ms" in result
        assert "Total" in result

    def test_format_latency_without_rerank(self) -> None:
        """Test latency formatting without reranking."""
        from demo.app import format_latency
        
        latency = {
            "retrieve_ms": 50.0,
            "rerank_ms": 0.0,
            "generate_ms": 4000.0,
            "total_ms": 4050.0,
        }
        
        result = format_latency(latency, show_rerank=False)
        
        assert "Retrieval" in result
        assert "Generation" in result
        assert "Reranking" not in result

    def test_format_config_summary(self) -> None:
        """Test configuration summary formatting."""
        from demo.app import format_config_summary
        
        config = {
            "retrieval": {"mode": "mmr", "k": 10, "fetch_k": 50},
            "rerank": {"enabled": True, "model": "cross-encoder/test", "final_k": 8},
        }
        
        result = format_config_summary(config, "Test Label")
        
        assert "Test Label" in result
        assert "mmr" in result
        assert "10" in result
        assert "Enabled" in result

    def test_disclaimer_html_exists(self) -> None:
        """Test medical disclaimer HTML is present."""
        from demo.app import DISCLAIMER_HTML
        
        assert "MEDICAL DISCLAIMER" in DISCLAIMER_HTML.upper() or "Medical Disclaimer" in DISCLAIMER_HTML
        assert "NOT" in DISCLAIMER_HTML
        assert "research" in DISCLAIMER_HTML.lower() or "educational" in DISCLAIMER_HTML.lower()

    def test_demo_class_initialization(self) -> None:
        """Test BioRAGDemo can be initialized."""
        from demo.app import BioRAGDemo
        
        demo = BioRAGDemo()
        
        assert demo.config_path is not None
        assert demo.index_path is not None
        assert demo._baseline_pipeline is None  # Lazy loaded
        assert demo._optimized_pipeline is None  # Lazy loaded
        assert demo._single_pipeline is None  # Lazy loaded

    def test_demo_class_with_custom_paths(self) -> None:
        """Test BioRAGDemo with custom config and index paths."""
        from demo.app import BioRAGDemo
        
        demo = BioRAGDemo(
            config_path="/custom/config.yaml",
            index_path="/custom/index",
        )
        
        assert demo.config_path == "/custom/config.yaml"
        assert demo.index_path == "/custom/index"

    def test_answer_question_single_empty_input(self) -> None:
        """Test single question answering with empty input."""
        from demo.app import BioRAGDemo
        
        demo = BioRAGDemo()
        
        result = demo.answer_question_single("", "auto")
        
        assert len(result) == 4
        assert "Please enter a question" in result[0]

    def test_answer_question_single_whitespace_only(self) -> None:
        """Test single question answering with whitespace-only input."""
        from demo.app import BioRAGDemo
        
        demo = BioRAGDemo()
        
        result = demo.answer_question_single("   ", "auto")
        
        assert "Please enter a question" in result[0]

    def test_answer_question_comparison_empty_input(self) -> None:
        """Test comparison mode with empty input."""
        from demo.app import BioRAGDemo
        
        demo = BioRAGDemo()
        
        result = demo.answer_question_comparison("", "auto")
        
        assert len(result) == 10
        assert "Please enter a question" in result[0]
        assert "Please enter a question" in result[5]


class TestDemoFormatters:
    """Tests for demo formatting functions."""

    def test_format_chunk_truncates_long_text(self) -> None:
        """Test that long text is truncated."""
        from demo.app import format_chunk
        
        long_text = "A" * 500  # Very long text
        chunk = {
            "pmid": "12345678",
            "text": long_text,
            "score": 0.9,
            "rank": 1,
        }
        
        result = format_chunk(chunk, 0, show_rerank=False)
        
        # Should be truncated (250 chars max + "...")
        assert len(result) < len(long_text) + 100

    def test_format_citation_truncates_long_quote(self) -> None:
        """Test that long quotes are truncated."""
        from demo.app import format_citation
        
        long_quote = "B" * 200  # Very long quote
        citation = {
            "pmid": "12345678",
            "quote": long_quote,
        }
        
        result = format_citation(citation, 1)
        
        # Should be truncated (120 chars max + "...")
        assert "..." in result


class TestCreateDemo:
    """Tests for create_demo function."""

    def test_create_demo_returns_gradio_blocks_and_instance(self) -> None:
        """Test create_demo returns a Gradio Blocks object and demo instance."""
        import gradio as gr
        from demo.app import create_demo, BioRAGDemo
        
        interface, demo_instance = create_demo()
        
        assert isinstance(interface, gr.Blocks)
        assert isinstance(demo_instance, BioRAGDemo)

    def test_create_demo_with_custom_paths(self) -> None:
        """Test create_demo accepts custom paths."""
        import gradio as gr
        from demo.app import create_demo
        
        interface, demo_instance = create_demo(
            config_path="/custom/config.yaml",
            index_path="/custom/index",
        )
        
        assert isinstance(interface, gr.Blocks)
        assert demo_instance.config_path == "/custom/config.yaml"

    def test_demo_instance_has_theme_and_css(self) -> None:
        """Test demo instance provides theme and css for Gradio 6.0+."""
        from demo.app import BioRAGDemo
        import gradio as gr
        
        demo = BioRAGDemo()
        
        theme = demo.get_theme()
        css = demo.get_css()
        
        assert isinstance(theme, gr.themes.Base)
        assert isinstance(css, str)
        assert len(css) > 100  # CSS should have content


class TestDemoConfigs:
    """Tests for demo configuration constants."""

    def test_baseline_is_simpler_than_optimized(self) -> None:
        """Test baseline config is simpler than optimized."""
        from demo.app import BASELINE_CONFIG, OPTIMIZED_CONFIG
        
        # Baseline should have simpler retrieval
        assert BASELINE_CONFIG["retrieval"]["k"] < OPTIMIZED_CONFIG["retrieval"]["k"]
        assert BASELINE_CONFIG["retrieval"]["fetch_k"] < OPTIMIZED_CONFIG["retrieval"]["fetch_k"]
        
        # Baseline should have reranking disabled
        assert BASELINE_CONFIG["rerank"]["enabled"] is False
        assert OPTIMIZED_CONFIG["rerank"]["enabled"] is True

    def test_optimized_uses_mmr(self) -> None:
        """Test optimized config uses MMR retrieval."""
        from demo.app import OPTIMIZED_CONFIG
        
        assert OPTIMIZED_CONFIG["retrieval"]["mode"] == "mmr"
        assert "lambda_mult" in OPTIMIZED_CONFIG["retrieval"]

    def test_optimized_has_reranker_model(self) -> None:
        """Test optimized config specifies reranker model."""
        from demo.app import OPTIMIZED_CONFIG
        
        assert "model" in OPTIMIZED_CONFIG["rerank"]
        assert "cross-encoder" in OPTIMIZED_CONFIG["rerank"]["model"]

