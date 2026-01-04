"""Token-based chunker for BioRAG Bench."""

from __future__ import annotations

import re

from biorag.chunking.base import Chunker
from biorag.schemas.corpus import Chunk


class TokenChunker(Chunker):
    """
    Simple token-based chunker that splits on word boundaries.

    This is a baseline chunker that doesn't consider sentence structure.
    """

    def __init__(
        self,
        chunk_size: int = 350,
        chunk_overlap: int = 40,
    ) -> None:
        """
        Initialize token chunker.

        Args:
            chunk_size: Target number of tokens per chunk
            chunk_overlap: Overlap in tokens between consecutive chunks
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """
        Chunk text by token count.

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

        # Simple tokenization (split on whitespace and punctuation)
        tokens = self._tokenize(text)

        if not tokens:
            return []

        if len(tokens) <= self.chunk_size:
            # Text fits in a single chunk
            return [
                self._create_chunk(
                    text=text.strip(),
                    pmid=pmid,
                    chunk_index=0,
                    total_chunks=1,
                    start_char=0,
                    end_char=len(text),
                )
            ]

        # Split into overlapping chunks
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        total_chunks = (len(tokens) + step - 1) // step

        for idx in range(0, len(tokens), step):
            chunk_tokens = tokens[idx : idx + self.chunk_size]
            if not chunk_tokens:
                break

            chunk_text = " ".join(chunk_tokens)

            # Estimate character positions
            start_char = idx * 6  # Rough estimate
            end_char = start_char + len(chunk_text)

            chunk = self._create_chunk(
                text=chunk_text,
                pmid=pmid,
                chunk_index=len(chunks),
                total_chunks=total_chunks,
                start_char=start_char,
                end_char=end_char,
            )
            chunks.append(chunk)

            if idx + self.chunk_size >= len(tokens):
                break

        # Update total_chunks with actual count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _tokenize(self, text: str) -> list[str]:
        """Simple word tokenization."""
        # Split on whitespace and keep punctuation attached
        return re.findall(r"\S+", text)

