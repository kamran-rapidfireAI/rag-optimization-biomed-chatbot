"""Base chunker interface for BioRAG Bench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from biorag.schemas.corpus import Chunk, CorpusDocument


class Chunker(ABC):
    """Abstract base class for text chunkers."""

    def __init__(
        self,
        chunk_size: int = 350,
        chunk_overlap: int = 40,
    ) -> None:
        """
        Initialize chunker.

        Args:
            chunk_size: Target size for each chunk (in characters or tokens)
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk_text(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """
        Chunk a text string into smaller pieces.

        Args:
            text: Text to chunk
            metadata: Optional metadata to include in chunks

        Returns:
            List of Chunk objects
        """
        pass

    def chunk_document(self, document: CorpusDocument) -> list[Chunk]:
        """
        Chunk a corpus document.

        Args:
            document: CorpusDocument to chunk

        Returns:
            List of Chunk objects with document metadata
        """
        text = document.full_text()
        metadata = {"pmid": document.pmid, "is_gold": document.is_gold}
        return self.chunk_text(text, metadata)

    def chunk_documents(
        self, documents: Iterator[CorpusDocument] | list[CorpusDocument]
    ) -> Iterator[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: Iterable of CorpusDocument objects

        Yields:
            Chunk objects from all documents
        """
        for doc in documents:
            yield from self.chunk_document(doc)

    def _create_chunk(
        self,
        text: str,
        pmid: str,
        chunk_index: int,
        total_chunks: int,
        start_char: int = 0,
        end_char: int = 0,
        section: str | None = None,
    ) -> Chunk:
        """Create a Chunk object with standard ID format."""
        chunk_id = f"{pmid}_{chunk_index}"
        return Chunk(
            chunk_id=chunk_id,
            pmid=pmid,
            text=text,
            start_char=start_char,
            end_char=end_char,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            section=section,
        )

