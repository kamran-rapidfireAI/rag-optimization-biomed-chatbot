"""Metadata store for chunk information."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from biorag.schemas.corpus import Chunk
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


class MetadataStore:
    """SQLite-based metadata store for chunks."""

    def __init__(self, db_path: Path | str) -> None:
        """
        Initialize metadata store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                pmid TEXT NOT NULL,
                text TEXT NOT NULL,
                start_char INTEGER DEFAULT 0,
                end_char INTEGER DEFAULT 0,
                chunk_index INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 1,
                section TEXT,
                token_count INTEGER,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pmid ON chunks(pmid)
        """)

        conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def add_chunk(self, chunk: Chunk, metadata: dict | None = None) -> None:
        """
        Add a chunk to the store.

        Args:
            chunk: Chunk to add
            metadata: Optional additional metadata
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO chunks
            (chunk_id, pmid, text, start_char, end_char, chunk_index,
             total_chunks, section, token_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.pmid,
                chunk.text,
                chunk.start_char,
                chunk.end_char,
                chunk.chunk_index,
                chunk.total_chunks,
                chunk.section,
                chunk.token_count,
                json.dumps(metadata) if metadata else None,
            ),
        )
        conn.commit()

    def add_chunks(self, chunks: list[Chunk] | Iterator[Chunk]) -> int:
        """
        Add multiple chunks to the store.

        Args:
            chunks: Chunks to add

        Returns:
            Number of chunks added
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        count = 0
        for chunk in chunks:
            cursor.execute(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, pmid, text, start_char, end_char, chunk_index,
                 total_chunks, section, token_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.pmid,
                    chunk.text,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.chunk_index,
                    chunk.total_chunks,
                    chunk.section,
                    chunk.token_count,
                    None,
                ),
            )
            count += 1

        conn.commit()
        logger.info(f"Added {count} chunks to metadata store")
        return count

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """
        Get a chunk by ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Chunk if found, None otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()

        if row:
            return self._row_to_chunk(row)
        return None

    def get_chunks_by_pmid(self, pmid: str) -> list[Chunk]:
        """
        Get all chunks for a document.

        Args:
            pmid: PubMed ID

        Returns:
            List of chunks for the document
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM chunks WHERE pmid = ? ORDER BY chunk_index", (pmid,)
        )

        return [self._row_to_chunk(row) for row in cursor.fetchall()]

    def get_all_chunks(self) -> Iterator[Chunk]:
        """
        Iterate over all chunks.

        Yields:
            Chunk objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM chunks ORDER BY pmid, chunk_index")

        for row in cursor:
            yield self._row_to_chunk(row)

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        """Convert database row to Chunk object."""
        return Chunk(
            chunk_id=row["chunk_id"],
            pmid=row["pmid"],
            text=row["text"],
            start_char=row["start_char"],
            end_char=row["end_char"],
            chunk_index=row["chunk_index"],
            total_chunks=row["total_chunks"],
            section=row["section"],
            token_count=row["token_count"],
        )

    def count(self) -> int:
        """Return total number of chunks."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        return cursor.fetchone()[0]

    def count_by_pmid(self) -> dict[str, int]:
        """Return chunk counts by PMID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT pmid, COUNT(*) as count FROM chunks GROUP BY pmid")
        return dict(cursor.fetchall())

    def clear(self) -> None:
        """Clear all chunks from the store."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks")
        conn.commit()
        logger.info("Cleared metadata store")

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

