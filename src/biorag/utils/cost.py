"""Cost control utilities for BioRAG Bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from biorag.utils.logging import get_logger

logger = get_logger(__name__)


# Token pricing per 1M tokens (as of 2024)
# Source: https://openai.com/pricing
MODEL_PRICING: dict[str, dict[str, float]] = {
    # GPT-4o models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    # GPT-4 Turbo
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    # GPT-4
    "gpt-4": {"input": 30.00, "output": 60.00},
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    # Embeddings
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
}


def get_model_pricing(model: str) -> dict[str, float]:
    """
    Get pricing for a model.

    Args:
        model: Model name

    Returns:
        Dict with 'input' and 'output' prices per 1M tokens
    """
    # Try exact match first
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    
    # Try prefix matching for versioned models
    for known_model in MODEL_PRICING:
        if model.startswith(known_model):
            return MODEL_PRICING[known_model]
    
    # Default to gpt-4o-mini pricing for unknown models
    logger.warning(f"Unknown model '{model}', using gpt-4o-mini pricing")
    return MODEL_PRICING["gpt-4o-mini"]


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Estimate cost for a request.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Estimated cost in USD
    """
    pricing = get_model_pricing(model)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


class BudgetExceededError(Exception):
    """Raised when budget limits are exceeded."""

    def __init__(
        self,
        message: str,
        limit_type: str,
        limit_value: float,
        current_value: float,
    ) -> None:
        super().__init__(message)
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.current_value = current_value


@dataclass
class CostTracker:
    """
    Tracks costs and enforces budget limits.
    
    Provides:
    - Token counting (input + output)
    - Budget guardrails (questions, tokens, USD)
    - Per-run cost reporting
    """

    # Budget limits
    max_questions: int | None = None
    max_total_tokens: int | None = None
    max_usd: float | None = None
    on_budget_exceeded: Literal["fail-fast", "skip"] = "fail-fast"

    # Model for cost estimation
    model: str = "gpt-4o-mini"

    # Current usage
    questions_processed: int = field(default=0, init=False)
    input_tokens: int = field(default=0, init=False)
    output_tokens: int = field(default=0, init=False)
    estimated_cost_usd: float = field(default=0.0, init=False)

    # Cache stats
    cache_hits: int = field(default=0, init=False)
    cache_misses: int = field(default=0, init=False)

    # Skipped due to budget
    questions_skipped: int = field(default=0, init=False)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate (0-1)."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def check_budget(self, raise_on_exceeded: bool = True) -> bool:
        """
        Check if budget limits allow another request.

        Args:
            raise_on_exceeded: Whether to raise exception on exceeded

        Returns:
            True if within budget

        Raises:
            BudgetExceededError: If budget exceeded and fail-fast mode
        """
        exceeded = None
        
        if self.max_questions is not None:
            if self.questions_processed >= self.max_questions:
                exceeded = BudgetExceededError(
                    f"Question limit reached: {self.questions_processed}/{self.max_questions}",
                    "questions",
                    self.max_questions,
                    self.questions_processed,
                )
        
        if self.max_total_tokens is not None:
            if self.total_tokens >= self.max_total_tokens:
                exceeded = BudgetExceededError(
                    f"Token limit reached: {self.total_tokens}/{self.max_total_tokens}",
                    "tokens",
                    self.max_total_tokens,
                    self.total_tokens,
                )
        
        if self.max_usd is not None:
            if self.estimated_cost_usd >= self.max_usd:
                exceeded = BudgetExceededError(
                    f"Cost limit reached: ${self.estimated_cost_usd:.4f}/${self.max_usd:.4f}",
                    "usd",
                    self.max_usd,
                    self.estimated_cost_usd,
                )
        
        if exceeded is not None:
            if raise_on_exceeded and self.on_budget_exceeded == "fail-fast":
                raise exceeded
            return False
        
        return True

    def should_skip(self) -> bool:
        """
        Check if we should skip the next question due to budget.

        Returns:
            True if should skip (budget exceeded and skip mode)
        """
        if not self.check_budget(raise_on_exceeded=False):
            if self.on_budget_exceeded == "skip":
                self.questions_skipped += 1
                return True
            # fail-fast mode - will raise on next add_request
        return False

    def add_request(
        self,
        input_tokens: int,
        output_tokens: int,
        is_cache_hit: bool = False,
    ) -> None:
        """
        Record a request.

        Args:
            input_tokens: Input tokens used
            output_tokens: Output tokens generated
            is_cache_hit: Whether this was a cache hit
        """
        self.questions_processed += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        
        if is_cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            # Only count cost for non-cached requests
            self.estimated_cost_usd += estimate_cost(
                self.model, input_tokens, output_tokens
            )
        
        # Check budget after adding
        self.check_budget(raise_on_exceeded=True)

    def get_report(self) -> dict:
        """
        Get cost report.

        Returns:
            Dict with usage statistics
        """
        return {
            "questions_processed": self.questions_processed,
            "questions_skipped": self.questions_skipped,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "model": self.model,
            "limits": {
                "max_questions": self.max_questions,
                "max_total_tokens": self.max_total_tokens,
                "max_usd": self.max_usd,
                "on_budget_exceeded": self.on_budget_exceeded,
            },
        }

    def reset(self) -> None:
        """Reset all counters."""
        self.questions_processed = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.estimated_cost_usd = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.questions_skipped = 0

    def __str__(self) -> str:
        """String representation with summary."""
        return (
            f"CostTracker("
            f"questions={self.questions_processed}, "
            f"tokens={self.total_tokens}, "
            f"cost=${self.estimated_cost_usd:.4f}, "
            f"cache_hit_rate={self.cache_hit_rate:.1%})"
        )


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Estimate token count for text.
    
    Uses tiktoken for accurate counting when available,
    falls back to character-based estimation.

    Args:
        text: Text to count tokens for
        model: Model for tokenizer selection

    Returns:
        Estimated token count
    """
    try:
        import tiktoken
        
        # Map model to encoding
        if "gpt-4o" in model or "gpt-4-turbo" in model:
            encoding_name = "o200k_base"
        elif "gpt-4" in model or "gpt-3.5" in model:
            encoding_name = "cl100k_base"
        else:
            encoding_name = "cl100k_base"
        
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except ImportError:
        # Fallback: ~4 chars per token for English
        return len(text) // 4


def count_tokens_messages(
    messages: list[dict[str, str]],
    model: str = "gpt-4o-mini",
) -> int:
    """
    Count tokens for chat messages.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model for tokenizer

    Returns:
        Total token count
    """
    total = 0
    for msg in messages:
        # Each message has overhead for role and formatting
        total += 4  # <|start|>role\n...<|end|>
        total += count_tokens(msg.get("content", ""), model)
        total += count_tokens(msg.get("role", "user"), model)
    
    # Add priming tokens
    total += 3
    
    return total










