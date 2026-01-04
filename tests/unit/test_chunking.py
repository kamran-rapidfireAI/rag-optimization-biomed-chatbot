"""Unit tests for chunking modules."""

from __future__ import annotations

import pytest

from biorag.chunking.base import Chunker
from biorag.chunking.recursive import RecursiveChunker
from biorag.chunking.token import TokenChunker
from biorag.schemas.corpus import Chunk, CorpusDocument


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    @pytest.fixture
    def chunker(self) -> RecursiveChunker:
        """Create a chunker instance."""
        return RecursiveChunker(chunk_size=100, chunk_overlap=20)

    def test_chunk_short_text(self, chunker: RecursiveChunker) -> None:
        """Test chunking short text that fits in one chunk."""
        text = "This is a short text."
        chunks = chunker.chunk_text(text, metadata={"pmid": "12345"})

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].pmid == "12345"
        assert chunks[0].chunk_index == 0

    def test_chunk_long_text(self, chunker: RecursiveChunker) -> None:
        """Test chunking longer text into multiple chunks."""
        text = "This is a sentence. " * 20  # ~400 chars
        chunks = chunker.chunk_text(text, metadata={"pmid": "12345"})

        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.pmid == "12345"
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)

    def test_chunk_empty_text(self, chunker: RecursiveChunker) -> None:
        """Test chunking empty text."""
        chunks = chunker.chunk_text("", metadata={"pmid": "12345"})
        assert len(chunks) == 0

    def test_chunk_ids_are_unique(self, chunker: RecursiveChunker) -> None:
        """Test that chunk IDs are unique."""
        text = "Long text. " * 50
        chunks = chunker.chunk_text(text, metadata={"pmid": "12345"})

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_document(self, chunker: RecursiveChunker) -> None:
        """Test chunking a CorpusDocument."""
        doc = CorpusDocument(
            pmid="12345",
            title="Test Title",
            abstract="This is a test abstract. " * 10,
        )
        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 1
        assert all(c.pmid == "12345" for c in chunks)


class TestTokenChunker:
    """Tests for TokenChunker."""

    @pytest.fixture
    def chunker(self) -> TokenChunker:
        """Create a chunker instance."""
        return TokenChunker(chunk_size=20, chunk_overlap=5)

    def test_chunk_short_text(self, chunker: TokenChunker) -> None:
        """Test chunking short text."""
        text = "This is a short text with few words."
        chunks = chunker.chunk_text(text, metadata={"pmid": "12345"})

        assert len(chunks) == 1
        assert chunks[0].pmid == "12345"

    def test_chunk_long_text(self, chunker: TokenChunker) -> None:
        """Test chunking longer text."""
        text = " ".join([f"word{i}" for i in range(100)])  # 100 words
        chunks = chunker.chunk_text(text, metadata={"pmid": "12345"})

        assert len(chunks) > 1

    def test_chunk_empty_text(self, chunker: TokenChunker) -> None:
        """Test chunking empty text."""
        chunks = chunker.chunk_text("", metadata={"pmid": "12345"})
        assert len(chunks) == 0

    def test_chunk_overlap(self, chunker: TokenChunker) -> None:
        """Test that chunks have overlap."""
        text = " ".join([f"word{i}" for i in range(50)])
        chunks = chunker.chunk_text(text, metadata={"pmid": "12345"})

        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            words_first = set(chunks[0].text.split())
            words_second = set(chunks[1].text.split())
            # There should be some overlap
            assert len(words_first & words_second) > 0


class TestChunkerInterface:
    """Tests for Chunker interface."""

    def test_chunk_documents_iterator(self) -> None:
        """Test chunking multiple documents."""
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)

        docs = [
            CorpusDocument(pmid="111", abstract="First abstract. " * 5),
            CorpusDocument(pmid="222", abstract="Second abstract. " * 5),
        ]

        all_chunks = list(chunker.chunk_documents(docs))

        # Should have chunks from both documents
        pmids = {c.pmid for c in all_chunks}
        assert "111" in pmids
        assert "222" in pmids

