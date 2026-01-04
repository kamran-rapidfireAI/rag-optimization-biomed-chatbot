"""Recursive character text splitter for BioRAG Bench."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from biorag.chunking.base import Chunker
from biorag.schemas.corpus import Chunk


class RecursiveChunker(Chunker):
    """
    Recursive character-based chunker using LangChain's RecursiveCharacterTextSplitter.

    This chunker is sentence-aware and tries to split on natural boundaries.
    """

    # Default separators optimized for biomedical text
    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraph breaks
        "\n",  # Line breaks
        ". ",  # Sentence endings
        "? ",  # Question marks
        "! ",  # Exclamation marks
        "; ",  # Semicolons
        ", ",  # Commas
        " ",  # Spaces
        "",  # Character level (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = 350,
        chunk_overlap: int = 40,
        separators: list[str] | None = None,
        keep_separator: bool = True,
        strip_whitespace: bool = True,
    ) -> None:
        """
        Initialize recursive chunker.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            separators: Custom list of separators (defaults to sentence-aware)
            keep_separator: Whether to keep separators in chunks
            strip_whitespace: Whether to strip whitespace from chunks
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.separators = separators or self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            keep_separator=keep_separator,
            strip_whitespace=strip_whitespace,
            length_function=len,
        )

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """
        Chunk text using recursive character splitting.

        Args:
            text: Text to chunk
            metadata: Optional metadata including 'pmid'

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        pmid = metadata.get("pmid", "unknown")

        # Use LangChain splitter
        texts = self._splitter.split_text(text)

        if not texts:
            return []

        # Create chunks with position tracking
        chunks = []
        current_pos = 0

        for idx, chunk_text in enumerate(texts):
            # Find the actual position in the original text
            try:
                start_pos = text.index(chunk_text, current_pos)
            except ValueError:
                # If exact match not found (due to stripping), estimate position
                start_pos = current_pos

            end_pos = start_pos + len(chunk_text)
            current_pos = start_pos + len(chunk_text) - self.chunk_overlap

            chunk = self._create_chunk(
                text=chunk_text,
                pmid=pmid,
                chunk_index=idx,
                total_chunks=len(texts),
                start_char=start_pos,
                end_char=end_pos,
            )
            chunks.append(chunk)

        return chunks

