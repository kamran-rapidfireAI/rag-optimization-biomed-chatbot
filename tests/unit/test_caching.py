"""Unit tests for LLM caching utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from biorag.utils.caching import LLMCache


class TestLLMCache:
    """Tests for LLMCache class."""

    @pytest.fixture
    def cache(self, tmp_path: Path) -> LLMCache:
        """Create a cache instance for testing."""
        return LLMCache(cache_dir=tmp_path / "cache")

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test that init creates cache directory."""
        cache_dir = tmp_path / "new_cache"
        
        LLMCache(cache_dir=cache_dir)
        
        assert cache_dir.exists()

    def test_compute_cache_key_deterministic(self) -> None:
        """Test cache key is deterministic."""
        key1 = LLMCache.compute_cache_key(
            model="gpt-4o-mini",
            prompt="What is the answer?",
            template_hash="abc123",
            temperature=0.0,
            max_tokens=350,
        )
        
        key2 = LLMCache.compute_cache_key(
            model="gpt-4o-mini",
            prompt="What is the answer?",
            template_hash="abc123",
            temperature=0.0,
            max_tokens=350,
        )
        
        assert key1 == key2
        assert len(key1) == 64  # SHA256

    def test_compute_cache_key_different_inputs(self) -> None:
        """Test different inputs produce different keys."""
        key1 = LLMCache.compute_cache_key(
            model="gpt-4o-mini",
            prompt="Prompt 1",
            temperature=0.0,
            max_tokens=350,
        )
        
        key2 = LLMCache.compute_cache_key(
            model="gpt-4o-mini",
            prompt="Prompt 2",
            temperature=0.0,
            max_tokens=350,
        )
        
        assert key1 != key2

    def test_compute_cache_key_temperature_matters(self) -> None:
        """Test that temperature affects cache key."""
        key1 = LLMCache.compute_cache_key(
            model="gpt-4o-mini",
            prompt="Same prompt",
            temperature=0.0,
            max_tokens=350,
        )
        
        key2 = LLMCache.compute_cache_key(
            model="gpt-4o-mini",
            prompt="Same prompt",
            temperature=0.7,
            max_tokens=350,
        )
        
        assert key1 != key2

    def test_get_miss(self, cache: LLMCache) -> None:
        """Test cache miss returns None."""
        result = cache.get("nonexistent_key")
        
        assert result is None
        assert cache._misses == 1

    def test_set_and_get(self, cache: LLMCache) -> None:
        """Test setting and getting cached value."""
        cache_key = "test_key"
        response = {"answer": "Test answer", "citations": []}
        
        cache.set(
            cache_key=cache_key,
            response=response,
            model="gpt-4o-mini",
            prompt_hash="prompt123",
            template_hash="template456",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500.0,
        )
        
        result = cache.get(cache_key)
        
        assert result is not None
        assert result["response"]["answer"] == "Test answer"
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["cache_hit"] is True

    def test_get_updates_access_stats(self, cache: LLMCache) -> None:
        """Test that get updates access statistics."""
        cache_key = "test_key"
        cache.set(
            cache_key=cache_key,
            response={"answer": "Test"},
            model="gpt-4o-mini",
            prompt_hash="p",
            input_tokens=0,
            output_tokens=0,
        )
        
        # First access
        cache.get(cache_key)
        # Second access
        cache.get(cache_key)
        
        assert cache._hits == 2

    def test_invalidate(self, cache: LLMCache) -> None:
        """Test invalidating a cache entry."""
        cache_key = "to_delete"
        cache.set(
            cache_key=cache_key,
            response={"answer": "Delete me"},
            model="gpt-4o-mini",
            prompt_hash="p",
            input_tokens=0,
            output_tokens=0,
        )
        
        # Verify it exists
        assert cache.get(cache_key) is not None
        
        # Invalidate
        result = cache.invalidate(cache_key)
        
        assert result is True
        assert cache.get(cache_key) is None

    def test_invalidate_nonexistent(self, cache: LLMCache) -> None:
        """Test invalidating non-existent key."""
        result = cache.invalidate("nonexistent")
        
        assert result is False

    def test_invalidate_model(self, cache: LLMCache) -> None:
        """Test invalidating all entries for a model."""
        # Add entries for different models
        cache.set("key1", {"a": 1}, "model-a", "p1", input_tokens=0, output_tokens=0)
        cache.set("key2", {"a": 2}, "model-a", "p2", input_tokens=0, output_tokens=0)
        cache.set("key3", {"b": 1}, "model-b", "p3", input_tokens=0, output_tokens=0)
        
        # Invalidate model-a
        count = cache.invalidate_model("model-a")
        
        assert count == 2
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is not None  # model-b still exists

    def test_clear(self, cache: LLMCache) -> None:
        """Test clearing all cache entries."""
        cache.set("key1", {"a": 1}, "model", "p", input_tokens=0, output_tokens=0)
        cache.set("key2", {"a": 2}, "model", "p", input_tokens=0, output_tokens=0)
        
        count = cache.clear()
        
        assert count == 2
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_stats(self, cache: LLMCache) -> None:
        """Test getting cache statistics."""
        cache.set("key1", {"a": 1}, "model", "p", input_tokens=100, output_tokens=50)
        cache.set("key2", {"a": 2}, "model", "p", input_tokens=200, output_tokens=100)
        
        cache.get("key1")  # Hit
        cache.get("key2")  # Hit
        cache.get("nonexistent")  # Miss
        
        stats = cache.stats
        
        assert stats["total_entries"] == 2
        assert stats["total_input_tokens"] == 300
        assert stats["total_output_tokens"] == 150
        assert stats["session_hits"] == 2
        assert stats["session_misses"] == 1
        assert stats["session_hit_rate"] == pytest.approx(2/3)

    def test_reset_session_stats(self, cache: LLMCache) -> None:
        """Test resetting session statistics."""
        cache.set("key1", {"a": 1}, "model", "p", input_tokens=0, output_tokens=0)
        cache.get("key1")
        cache.get("nonexistent")
        
        cache.reset_session_stats()
        
        assert cache._hits == 0
        assert cache._misses == 0

    def test_persistence(self, tmp_path: Path) -> None:
        """Test cache persists across instances."""
        cache_dir = tmp_path / "persistent_cache"
        
        # First instance - write
        cache1 = LLMCache(cache_dir=cache_dir)
        cache1.set("persistent_key", {"data": "value"}, "model", "p", input_tokens=0, output_tokens=0)
        cache1.close()
        
        # Second instance - read
        cache2 = LLMCache(cache_dir=cache_dir)
        result = cache2.get("persistent_key")
        
        assert result is not None
        assert result["response"]["data"] == "value"


class TestLLMCacheEdgeCases:
    """Edge case tests for LLMCache."""

    def test_complex_response(self, tmp_path: Path) -> None:
        """Test caching complex nested response."""
        cache = LLMCache(cache_dir=tmp_path)
        
        complex_response = {
            "answer": "Complex answer",
            "citations": [
                {"pmid": "123", "chunk_id": "123_0", "quote": "Quote 1"},
                {"pmid": "456", "chunk_id": "456_0", "quote": "Quote 2"},
            ],
            "nested": {
                "level1": {
                    "level2": ["a", "b", "c"]
                }
            }
        }
        
        cache.set("complex", complex_response, "model", "p", input_tokens=0, output_tokens=0)
        result = cache.get("complex")
        
        assert result["response"]["citations"][0]["pmid"] == "123"
        assert result["response"]["nested"]["level1"]["level2"] == ["a", "b", "c"]

    def test_unicode_content(self, tmp_path: Path) -> None:
        """Test caching content with unicode."""
        cache = LLMCache(cache_dir=tmp_path)
        
        response = {"answer": "Ответ на русском языке 中文 日本語 🧬"}
        
        cache.set("unicode", response, "model", "p", input_tokens=0, output_tokens=0)
        result = cache.get("unicode")
        
        assert result["response"]["answer"] == response["answer"]

    def test_empty_response(self, tmp_path: Path) -> None:
        """Test caching empty response."""
        cache = LLMCache(cache_dir=tmp_path)
        
        cache.set("empty", {}, "model", "p", input_tokens=0, output_tokens=0)
        result = cache.get("empty")
        
        assert result["response"] == {}

    def test_stats_empty_cache(self, tmp_path: Path) -> None:
        """Test stats on empty cache."""
        cache = LLMCache(cache_dir=tmp_path)
        
        stats = cache.stats
        
        assert stats["total_entries"] == 0
        assert stats["total_input_tokens"] == 0
        assert stats["session_hit_rate"] == 0.0





