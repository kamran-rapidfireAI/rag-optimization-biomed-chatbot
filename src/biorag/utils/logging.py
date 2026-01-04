"""Structured logging utilities for BioRAG Bench pipeline."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

# Global console for rich output
console = Console()

# Store configured loggers to avoid duplicate handlers
_configured_loggers: set[str] = set()


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter for production environments."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields from the record
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add source location for debug
        if record.levelno <= logging.DEBUG:
            log_data["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        return json.dumps(log_data, default=str)


class PipelineLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds pipeline stage context to log messages."""

    def __init__(self, logger: logging.Logger, stage: str | None = None) -> None:
        """Initialize with optional pipeline stage context."""
        super().__init__(logger, {"stage": stage})

    def process(
        self, msg: str, kwargs: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Add stage context to log messages."""
        extra = kwargs.get("extra", {})
        if self.extra.get("stage"):
            extra["stage"] = self.extra["stage"]
        kwargs["extra"] = extra
        return msg, kwargs

    def with_stage(self, stage: str) -> PipelineLoggerAdapter:
        """Create a new adapter with the specified stage."""
        return PipelineLoggerAdapter(self.logger, stage)

    def log_retrieval(
        self,
        query: str,
        num_results: int,
        latency_ms: float,
        **kwargs: Any,
    ) -> None:
        """Log retrieval stage metrics."""
        self.info(
            f"Retrieved {num_results} chunks in {latency_ms:.1f}ms",
            extra={
                "extra_data": {
                    "stage": "retrieve",
                    "query_preview": query[:100],
                    "num_results": num_results,
                    "latency_ms": latency_ms,
                    **kwargs,
                }
            },
        )

    def log_rerank(
        self,
        input_count: int,
        output_count: int,
        latency_ms: float,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log reranking stage metrics."""
        self.info(
            f"Reranked {input_count} → {output_count} chunks in {latency_ms:.1f}ms",
            extra={
                "extra_data": {
                    "stage": "rerank",
                    "input_count": input_count,
                    "output_count": output_count,
                    "latency_ms": latency_ms,
                    "model": model,
                    **kwargs,
                }
            },
        )

    def log_generation(
        self,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model: str | None = None,
        abstained: bool = False,
        **kwargs: Any,
    ) -> None:
        """Log generation stage metrics."""
        status = "abstained" if abstained else "generated"
        self.info(
            f"Generation {status}: {input_tokens} in / {output_tokens} out in {latency_ms:.1f}ms",
            extra={
                "extra_data": {
                    "stage": "generate",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": latency_ms,
                    "model": model,
                    "abstained": abstained,
                    **kwargs,
                }
            },
        )

    def log_eval_progress(
        self,
        current: int,
        total: int,
        question_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log evaluation progress."""
        pct = (current / total) * 100 if total > 0 else 0
        self.info(
            f"Evaluation progress: {current}/{total} ({pct:.1f}%)",
            extra={
                "extra_data": {
                    "stage": "eval",
                    "current": current,
                    "total": total,
                    "question_id": question_id,
                    **kwargs,
                }
            },
        )


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: str | None = None,
) -> None:
    """
    Set up logging configuration for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Whether to use JSON structured logging
        log_file: Optional file path to write logs to
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if json_format:
        # JSON structured logging for production
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
    else:
        # Rich console logging for development
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger.addHandler(handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for logger_name in ["httpx", "httpcore", "openai", "urllib3", "asyncio"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str, stage: str | None = None) -> PipelineLoggerAdapter:
    """
    Get a logger instance with optional pipeline stage context.

    Args:
        name: Logger name (typically __name__)
        stage: Optional pipeline stage for context

    Returns:
        PipelineLoggerAdapter instance
    """
    logger = logging.getLogger(name)

    # Ensure logging is set up at least once
    if not _configured_loggers:
        setup_logging()

    _configured_loggers.add(name)

    return PipelineLoggerAdapter(logger, stage)

