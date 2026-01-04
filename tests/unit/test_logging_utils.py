"""Unit tests for logging utilities - focused on actual logging behavior."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from biorag.utils.logging import (
    PipelineLoggerAdapter,
    StructuredFormatter,
    get_logger,
    setup_logging,
)


class TestStructuredFormatter:
    """Tests for JSON structured log formatting."""
    
    def test_formats_as_valid_json(self) -> None:
        """Output should be parseable JSON with required fields."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="biorag.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Processing %d items",
            args=(10,),
            exc_info=None,
        )
        
        output = formatter.format(record)
        parsed = json.loads(output)
        
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "biorag.test"
        assert parsed["message"] == "Processing 10 items"
        assert "timestamp" in parsed

    def test_includes_extra_data_from_record(self) -> None:
        """Extra data should be merged into output."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0, msg="msg", args=(), exc_info=None
        )
        record.extra_data = {"stage": "retrieve", "latency_ms": 50.5}
        
        output = formatter.format(record)
        parsed = json.loads(output)
        
        assert parsed["stage"] == "retrieve"
        assert parsed["latency_ms"] == 50.5

    def test_debug_level_includes_source_location(self) -> None:
        """DEBUG should include file/line/function for debugging."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.DEBUG,
            pathname="/app/module.py", lineno=123,
            msg="Debug info", args=(), exc_info=None
        )
        record.funcName = "process_data"
        
        output = formatter.format(record)
        parsed = json.loads(output)
        
        assert parsed["source"]["file"] == "/app/module.py"
        assert parsed["source"]["line"] == 123
        assert parsed["source"]["function"] == "process_data"

    def test_exception_info_included(self) -> None:
        """Exceptions should be captured in output."""
        formatter = StructuredFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test", level=logging.ERROR,
                pathname="", lineno=0, msg="Error occurred",
                args=(), exc_info=sys.exc_info()
            )
        
        output = formatter.format(record)
        parsed = json.loads(output)
        
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]
        assert "Test error" in parsed["exception"]


class TestPipelineLoggerAdapter:
    """Tests for pipeline stage logging."""
    
    def test_log_retrieval_formats_correctly(self, caplog: pytest.LogCaptureFixture) -> None:
        """log_retrieval should include query preview and metrics."""
        logger = get_logger("test")
        
        with caplog.at_level(logging.INFO):
            logger.log_retrieval(
                query="What is the treatment for cancer?",
                num_results=10,
                latency_ms=45.3,
            )
        
        assert "Retrieved 10 chunks" in caplog.text
        assert "45.3ms" in caplog.text

    def test_log_rerank_formats_correctly(self, caplog: pytest.LogCaptureFixture) -> None:
        """log_rerank should show before/after counts."""
        logger = get_logger("test")
        
        with caplog.at_level(logging.INFO):
            logger.log_rerank(
                input_count=50,
                output_count=8,
                latency_ms=120.0,
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            )
        
        assert "50 → 8" in caplog.text
        assert "120.0ms" in caplog.text

    def test_log_generation_shows_tokens(self, caplog: pytest.LogCaptureFixture) -> None:
        """log_generation should show token counts."""
        logger = get_logger("test")
        
        with caplog.at_level(logging.INFO):
            logger.log_generation(
                input_tokens=1500,
                output_tokens=250,
                latency_ms=800.0,
            )
        
        assert "1500 in" in caplog.text
        assert "250 out" in caplog.text

    def test_log_generation_indicates_abstention(self, caplog: pytest.LogCaptureFixture) -> None:
        """Abstention should be clearly indicated."""
        logger = get_logger("test")
        
        with caplog.at_level(logging.INFO):
            logger.log_generation(
                input_tokens=500,
                output_tokens=50,
                latency_ms=200.0,
                abstained=True,
            )
        
        assert "abstained" in caplog.text

    def test_with_stage_creates_new_adapter(self) -> None:
        """with_stage should create independent adapter."""
        original = get_logger("test", stage="retrieve")
        new = original.with_stage("generate")
        
        assert original.extra["stage"] == "retrieve"
        assert new.extra["stage"] == "generate"


class TestSetupLogging:
    """Tests for logging configuration."""
    
    def test_json_format_uses_structured_formatter(self) -> None:
        """JSON mode should use StructuredFormatter."""
        setup_logging(json_format=True, level="INFO")
        
        root = logging.getLogger()
        has_structured = any(
            isinstance(h.formatter, StructuredFormatter)
            for h in root.handlers
        )
        assert has_structured

    def test_log_file_creates_file(self, tmp_path: Path) -> None:
        """File logging should create the log file."""
        log_file = tmp_path / "app.log"
        setup_logging(log_file=str(log_file))
        
        logger = logging.getLogger("test_file")
        logger.warning("Test warning message")
        
        assert log_file.exists()

    def test_log_level_is_respected(self) -> None:
        """Configured level should be applied."""
        setup_logging(level="WARNING")
        
        root = logging.getLogger()
        assert root.level == logging.WARNING


class TestGetLogger:
    """Tests for get_logger factory."""
    
    def test_returns_pipeline_adapter(self) -> None:
        """Should return adapter with pipeline logging methods."""
        logger = get_logger("test.module")
        
        assert isinstance(logger, PipelineLoggerAdapter)
        assert hasattr(logger, "log_retrieval")
        assert hasattr(logger, "log_rerank")
        assert hasattr(logger, "log_generation")

    def test_stage_is_included(self) -> None:
        """Stage should be stored in adapter."""
        logger = get_logger("test", stage="eval")
        assert logger.extra["stage"] == "eval"
