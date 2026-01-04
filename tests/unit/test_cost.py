"""Unit tests for cost control utilities."""

from __future__ import annotations

import pytest

from biorag.utils.cost import (
    BudgetExceededError,
    CostTracker,
    count_tokens,
    count_tokens_messages,
    estimate_cost,
    get_model_pricing,
)


class TestModelPricing:
    """Tests for model pricing functions."""

    def test_get_known_model_pricing(self) -> None:
        """Test getting pricing for known models."""
        pricing = get_model_pricing("gpt-4o-mini")
        
        assert pricing["input"] == 0.15
        assert pricing["output"] == 0.60

    def test_get_gpt4o_pricing(self) -> None:
        """Test GPT-4o pricing."""
        pricing = get_model_pricing("gpt-4o")
        
        assert pricing["input"] == 2.50
        assert pricing["output"] == 10.00

    def test_get_unknown_model_pricing(self) -> None:
        """Test fallback for unknown models."""
        pricing = get_model_pricing("unknown-model")
        
        # Should default to gpt-4o-mini pricing
        assert pricing["input"] == 0.15
        assert pricing["output"] == 0.60

    def test_get_versioned_model_pricing(self) -> None:
        """Test pricing for versioned model names."""
        pricing = get_model_pricing("gpt-4o-2024-11-20")
        
        assert pricing["input"] == 2.50
        assert pricing["output"] == 10.00

    def test_embedding_pricing(self) -> None:
        """Test embedding model pricing."""
        pricing = get_model_pricing("text-embedding-3-large")
        
        assert pricing["input"] == 0.13
        assert pricing["output"] == 0.0


class TestEstimateCost:
    """Tests for cost estimation."""

    def test_estimate_cost_basic(self) -> None:
        """Test basic cost estimation."""
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        cost = estimate_cost("gpt-4o-mini", 1000, 500)
        
        # 1000 * 0.15/1M + 500 * 0.60/1M
        expected = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
        assert abs(cost - expected) < 0.0001

    def test_estimate_cost_large_usage(self) -> None:
        """Test cost estimation with larger token counts."""
        cost = estimate_cost("gpt-4o", 100_000, 50_000)
        
        # 100k * 2.50/1M + 50k * 10.00/1M
        expected = (100_000 / 1_000_000) * 2.50 + (50_000 / 1_000_000) * 10.00
        assert abs(cost - expected) < 0.0001

    def test_estimate_cost_embeddings(self) -> None:
        """Test cost for embedding models (output = 0)."""
        cost = estimate_cost("text-embedding-3-large", 10_000, 0)
        
        expected = (10_000 / 1_000_000) * 0.13
        assert abs(cost - expected) < 0.0001


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_initial_state(self) -> None:
        """Test initial tracker state."""
        tracker = CostTracker()
        
        assert tracker.questions_processed == 0
        assert tracker.input_tokens == 0
        assert tracker.output_tokens == 0
        assert tracker.total_tokens == 0
        assert tracker.estimated_cost_usd == 0.0
        assert tracker.cache_hit_rate == 0.0

    def test_add_request(self) -> None:
        """Test adding a request."""
        tracker = CostTracker(model="gpt-4o-mini")
        
        tracker.add_request(1000, 500, is_cache_hit=False)
        
        assert tracker.questions_processed == 1
        assert tracker.input_tokens == 1000
        assert tracker.output_tokens == 500
        assert tracker.total_tokens == 1500
        assert tracker.cache_misses == 1
        assert tracker.cache_hits == 0
        assert tracker.estimated_cost_usd > 0

    def test_add_cached_request(self) -> None:
        """Test adding a cached request (no cost)."""
        tracker = CostTracker(model="gpt-4o-mini")
        
        tracker.add_request(1000, 500, is_cache_hit=True)
        
        assert tracker.questions_processed == 1
        assert tracker.cache_hits == 1
        assert tracker.estimated_cost_usd == 0.0  # Cache hits don't cost

    def test_cache_hit_rate(self) -> None:
        """Test cache hit rate calculation."""
        tracker = CostTracker()
        
        tracker.add_request(100, 50, is_cache_hit=True)
        tracker.add_request(100, 50, is_cache_hit=True)
        tracker.add_request(100, 50, is_cache_hit=False)
        tracker.add_request(100, 50, is_cache_hit=False)
        
        assert tracker.cache_hit_rate == 0.5

    def test_budget_limit_questions(self) -> None:
        """Test question limit enforcement."""
        tracker = CostTracker(max_questions=2, on_budget_exceeded="fail-fast")
        
        tracker.add_request(100, 50)  # 1 question
        
        # Second request exceeds budget (check happens after adding)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.add_request(100, 50)  # 2 questions, then check raises
        
        assert exc_info.value.limit_type == "questions"
        assert exc_info.value.limit_value == 2

    def test_budget_limit_tokens(self) -> None:
        """Test token limit enforcement."""
        tracker = CostTracker(max_total_tokens=1000, on_budget_exceeded="fail-fast")
        
        tracker.add_request(400, 200)  # 600 tokens
        
        # Second request exceeds token budget
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.add_request(400, 200)  # 1200 total, check after adding raises
        
        assert exc_info.value.limit_type == "tokens"

    def test_budget_limit_usd(self) -> None:
        """Test USD limit enforcement."""
        tracker = CostTracker(max_usd=0.001, on_budget_exceeded="fail-fast", model="gpt-4o-mini")
        
        # First request should exceed $0.001 and raise after adding
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.add_request(10000, 5000)  # Will cost more than $0.001
        
        assert exc_info.value.limit_type == "usd"

    def test_budget_skip_mode(self) -> None:
        """Test skip mode when budget exceeded."""
        tracker = CostTracker(max_questions=2, on_budget_exceeded="skip")
        
        tracker.add_request(100, 50)
        tracker.add_request(100, 50)
        
        # Should not raise, but should indicate skip
        assert tracker.should_skip()
        assert tracker.questions_skipped == 1

    def test_check_budget_within_limits(self) -> None:
        """Test budget check when within limits."""
        tracker = CostTracker(
            max_questions=10,
            max_total_tokens=10000,
            max_usd=1.0,
        )
        
        tracker.add_request(100, 50)
        
        assert tracker.check_budget(raise_on_exceeded=False)

    def test_get_report(self) -> None:
        """Test getting cost report."""
        tracker = CostTracker(
            max_questions=100,
            model="gpt-4o-mini",
        )
        
        tracker.add_request(1000, 500, is_cache_hit=False)
        tracker.add_request(1000, 500, is_cache_hit=True)
        
        report = tracker.get_report()
        
        assert report["questions_processed"] == 2
        assert report["input_tokens"] == 2000
        assert report["output_tokens"] == 1000
        assert report["total_tokens"] == 3000
        assert report["cache_hits"] == 1
        assert report["cache_misses"] == 1
        assert report["cache_hit_rate"] == 0.5
        assert report["model"] == "gpt-4o-mini"
        assert report["limits"]["max_questions"] == 100

    def test_reset(self) -> None:
        """Test resetting tracker."""
        tracker = CostTracker()
        
        tracker.add_request(1000, 500)
        tracker.reset()
        
        assert tracker.questions_processed == 0
        assert tracker.total_tokens == 0
        assert tracker.estimated_cost_usd == 0.0

    def test_str_representation(self) -> None:
        """Test string representation."""
        tracker = CostTracker()
        tracker.add_request(1000, 500)
        
        s = str(tracker)
        
        assert "questions=1" in s
        assert "tokens=1500" in s


class TestCountTokens:
    """Tests for token counting functions."""

    def test_count_tokens_basic(self) -> None:
        """Test basic token counting."""
        text = "Hello, world! This is a test."
        
        count = count_tokens(text)
        
        # Should be a reasonable estimate
        assert count > 0
        assert count < len(text)

    def test_count_tokens_empty(self) -> None:
        """Test counting empty string."""
        count = count_tokens("")
        
        assert count == 0

    def test_count_tokens_long_text(self) -> None:
        """Test counting long text."""
        text = "word " * 1000
        
        count = count_tokens(text)
        
        # Should be roughly 1000 tokens (one per word)
        assert 500 < count < 2000

    def test_count_tokens_messages(self) -> None:
        """Test counting tokens in messages."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]
        
        count = count_tokens_messages(messages)
        
        # Should include overhead for roles and formatting
        assert count > 0


class TestBudgetExceededError:
    """Tests for BudgetExceededError exception."""

    def test_error_properties(self) -> None:
        """Test error has correct properties."""
        error = BudgetExceededError(
            "Budget exceeded",
            limit_type="tokens",
            limit_value=1000,
            current_value=1500,
        )
        
        assert error.limit_type == "tokens"
        assert error.limit_value == 1000
        assert error.current_value == 1500
        assert str(error) == "Budget exceeded"

