"""Unit tests for metadata store - focused on actual storage behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from biorag.indexing.metadata_store import MetadataStore
from biorag.schemas.corpus import Chunk


class TestMetadataStoreStorage:
    """Tests for storing and retrieving chunks."""
    
    @pytest.fixture
    def store(self, tmp_path: Path) -> MetadataStore:
        return MetadataStore(tmp_path / "test.db")

    def test_add_and_retrieve_chunk(self, store: MetadataStore) -> None:
        """Basic add/get should preserve all chunk fields."""
        chunk = Chunk(
            chunk_id="doc123_0",
            pmid="doc123",
            text="Important finding about biomarkers",
            start_char=0,
            end_char=35,
            chunk_index=0,
            total_chunks=3,
            section="abstract",
            token_count=6,
        )
        
        store.add_chunk(chunk)
        result = store.get_chunk("doc123_0")
        
        assert result is not None
        assert result.chunk_id == "doc123_0"
        assert result.pmid == "doc123"
        assert result.text == "Important finding about biomarkers"
        assert result.start_char == 0
        assert result.end_char == 35
        assert result.section == "abstract"
        assert result.token_count == 6

    def test_add_chunk_replaces_on_conflict(self, store: MetadataStore) -> None:
        """Adding chunk with same ID should replace, not duplicate."""
        store.add_chunk(Chunk(chunk_id="same_id", pmid="v1", text="Version 1"))
        store.add_chunk(Chunk(chunk_id="same_id", pmid="v2", text="Version 2"))
        
        assert store.count() == 1
        result = store.get_chunk("same_id")
        assert result.text == "Version 2"

    def test_add_chunks_batch(self, store: MetadataStore) -> None:
        """Batch add should be atomic and return count."""
        chunks = [
            Chunk(chunk_id=f"batch_{i}", pmid="doc", text=f"Chunk {i}")
            for i in range(100)
        ]
        
        count = store.add_chunks(chunks)
        
        assert count == 100
        assert store.count() == 100

    def test_get_nonexistent_returns_none(self, store: MetadataStore) -> None:
        """Missing chunk should return None, not raise."""
        assert store.get_chunk("does_not_exist") is None


class TestMetadataStoreQueries:
    """Tests for querying stored chunks."""
    
    @pytest.fixture
    def store(self, tmp_path: Path) -> MetadataStore:
        store = MetadataStore(tmp_path / "test.db")
        # Pre-populate with test data
        chunks = [
            Chunk(chunk_id="A_0", pmid="A", text="A first", chunk_index=0),
            Chunk(chunk_id="A_1", pmid="A", text="A second", chunk_index=1),
            Chunk(chunk_id="A_2", pmid="A", text="A third", chunk_index=2),
            Chunk(chunk_id="B_0", pmid="B", text="B only", chunk_index=0),
        ]
        store.add_chunks(chunks)
        return store

    def test_get_chunks_by_pmid_returns_ordered(self, store: MetadataStore) -> None:
        """Chunks for a document should be returned in chunk_index order."""
        chunks = store.get_chunks_by_pmid("A")
        
        assert len(chunks) == 3
        assert [c.chunk_index for c in chunks] == [0, 1, 2]

    def test_get_chunks_by_pmid_missing_returns_empty(self, store: MetadataStore) -> None:
        """Missing PMID should return empty list, not None."""
        chunks = store.get_chunks_by_pmid("nonexistent")
        assert chunks == []

    def test_count_by_pmid(self, store: MetadataStore) -> None:
        """Should count chunks per document."""
        counts = store.count_by_pmid()
        
        assert counts["A"] == 3
        assert counts["B"] == 1

    def test_get_all_chunks_iterates_all(self, store: MetadataStore) -> None:
        """Should iterate all chunks in store."""
        all_chunks = list(store.get_all_chunks())
        assert len(all_chunks) == 4


class TestMetadataStorePersistence:
    """Tests for data persistence across sessions."""
    
    def test_data_persists_after_close(self, tmp_path: Path) -> None:
        """Data should survive close and reopen."""
        db_path = tmp_path / "persistent.db"
        
        # Session 1: Write
        store1 = MetadataStore(db_path)
        store1.add_chunk(Chunk(chunk_id="persist_test", pmid="doc", text="Should persist"))
        store1.close()
        
        # Session 2: Read
        store2 = MetadataStore(db_path)
        result = store2.get_chunk("persist_test")
        
        assert result is not None
        assert result.text == "Should persist"
        store2.close()

    def test_clear_removes_all(self, tmp_path: Path) -> None:
        """Clear should remove all data."""
        store = MetadataStore(tmp_path / "test.db")
        store.add_chunks([
            Chunk(chunk_id=f"c_{i}", pmid="doc", text="text")
            for i in range(10)
        ])
        
        store.clear()
        
        assert store.count() == 0
        assert list(store.get_all_chunks()) == []
