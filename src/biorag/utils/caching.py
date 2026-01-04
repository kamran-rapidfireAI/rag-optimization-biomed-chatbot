"""LLM output caching for BioRAG Bench."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from biorag.utils.logging import get_logger

logger = get_logger(__name__)


class LLMCache:
    """
    Disk-based cache for LLM outputs.
    
    Uses SQLite for persistent storage, keyed by a stable hash of:
    - Model name
    - Prompt template version (hash)
    - Full rendered prompt
    - Decoding parameters (temperature, max_tokens, etc.)
    
    This prevents re-paying for identical evaluations during sweeps.
    """

    def __init__(
        self,
        cache_dir: Path | str = "data/cache",
        db_name: str = "llm_cache.db",
    ) -> None:
        """
        Initialize cache.

        Args:
            cache_dir: Directory for cache storage
            db_name: SQLite database filename
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / db_name
        
        # Thread-local connections for thread safety
        self._local = threading.local()
        
        # Initialize database schema
        self._init_db()
        
        # Stats
        self._hits = 0
        self._misses = 0

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                template_hash TEXT,
                prompt_hash TEXT NOT NULL,
                response TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                latency_ms REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_template
            ON llm_cache (model, template_hash)
        """)
        conn.commit()

    @staticmethod
    def compute_cache_key(
        model: str,
        prompt: str,
        template_hash: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 350,
        **kwargs: Any,
    ) -> str:
        """
        Compute stable cache key for LLM request.

        Args:
            model: Model name
            prompt: Full rendered prompt
            template_hash: Hash of prompt template
            temperature: Sampling temperature
            max_tokens: Max output tokens
            **kwargs: Additional decoding params

        Returns:
            SHA256 hash as cache key
        """
        # Build key components
        key_parts = {
            "model": model,
            "prompt": prompt,
            "template_hash": template_hash,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add any additional params that affect output
        for k, v in sorted(kwargs.items()):
            if k not in key_parts:
                key_parts[k] = v
        
        # Serialize deterministically
        key_str = json.dumps(key_parts, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        """
        Get cached response.

        Args:
            cache_key: Cache key from compute_cache_key()

        Returns:
            Cached response dict or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT response, input_tokens, output_tokens, latency_ms
            FROM llm_cache
            WHERE cache_key = ?
            """,
            (cache_key,),
        )
        row = cursor.fetchone()
        
        if row is None:
            self._misses += 1
            return None
        
        # Update access stats
        conn.execute(
            """
            UPDATE llm_cache
            SET accessed_at = ?, access_count = access_count + 1
            WHERE cache_key = ?
            """,
            (datetime.utcnow().isoformat(), cache_key),
        )
        conn.commit()
        
        self._hits += 1
        
        return {
            "response": json.loads(row["response"]),
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "latency_ms": row["latency_ms"],
            "cache_hit": True,
        }

    def set(
        self,
        cache_key: str,
        response: dict[str, Any],
        model: str,
        prompt_hash: str,
        template_hash: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Store response in cache.

        Args:
            cache_key: Cache key
            response: Response dict to cache
            model: Model name
            prompt_hash: Hash of the prompt
            template_hash: Hash of the template
            input_tokens: Input token count
            output_tokens: Output token count
            latency_ms: Response latency
        """
        now = datetime.utcnow().isoformat()
        conn = self._get_connection()
        
        conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache
            (cache_key, model, template_hash, prompt_hash, response,
             input_tokens, output_tokens, latency_ms, created_at, accessed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                model,
                template_hash,
                prompt_hash,
                json.dumps(response, default=str),
                input_tokens,
                output_tokens,
                latency_ms,
                now,
                now,
            ),
        )
        conn.commit()

    def invalidate(self, cache_key: str) -> bool:
        """
        Remove entry from cache.

        Args:
            cache_key: Cache key to remove

        Returns:
            True if entry was removed
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM llm_cache WHERE cache_key = ?",
            (cache_key,),
        )
        conn.commit()
        return cursor.rowcount > 0

    def invalidate_model(self, model: str) -> int:
        """
        Invalidate all cache entries for a model.

        Args:
            model: Model name

        Returns:
            Number of entries removed
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM llm_cache WHERE model = ?",
            (model,),
        )
        conn.commit()
        return cursor.rowcount

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries removed
        """
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM llm_cache")
        conn.commit()
        return cursor.rowcount

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) as count FROM llm_cache")
        total_entries = cursor.fetchone()["count"]
        
        cursor = conn.execute(
            "SELECT SUM(input_tokens) as inp, SUM(output_tokens) as out FROM llm_cache"
        )
        row = cursor.fetchone()
        
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "total_entries": total_entries,
            "total_input_tokens": row["inp"] or 0,
            "total_output_tokens": row["out"] or 0,
            "session_hits": self._hits,
            "session_misses": self._misses,
            "session_hit_rate": hit_rate,
        }

    def reset_session_stats(self) -> None:
        """Reset session hit/miss counters."""
        self._hits = 0
        self._misses = 0

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn

