#!/usr/bin/env python3
"""
Build a pre-optimized FAISS index for the demo.

This script creates a small but representative FAISS index for the demo,
using a subset of PubMedQA abstracts. The index is saved to the demo 
directory for deployment on HuggingFace Spaces.

Usage:
    python scripts/build_demo_index.py

Environment:
    OPENAI_API_KEY: Required for OpenAI embeddings

Output:
    demo/index/          - FAISS index directory
    demo/index/index.faiss
    demo/index/metadata.db
    demo/index/config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


def build_demo_index(
    output_dir: Path,
    num_documents: int = 500,
    chunk_size: int = 350,
    chunk_overlap: int = 40,
    embeddings_provider: str = "openai",
    embeddings_model: str = "text-embedding-3-large",
) -> None:
    """
    Build a demo-sized FAISS index.
    
    Args:
        output_dir: Directory to save the index
        num_documents: Number of documents to include
        chunk_size: Chunk size for splitting
        chunk_overlap: Chunk overlap for splitting
        embeddings_provider: Embeddings provider (openai or local)
        embeddings_model: Embeddings model name
    """
    from biorag.chunking.recursive import RecursiveChunker
    from biorag.data.pubmedqa_loader import PubMedQALoader
    from biorag.embeddings.local import LocalEmbedder
    from biorag.embeddings.openai import OpenAIEmbedder
    from biorag.indexing.faiss_store import FAISSStore
    from biorag.schemas.corpus import Chunk, CorpusDocument

    print("🧬 Building Demo FAISS Index")
    print("=" * 50)
    print(f"   Output: {output_dir}")
    print(f"   Documents: {num_documents}")
    print(f"   Chunk Size: {chunk_size}")
    print(f"   Embeddings: {embeddings_provider}/{embeddings_model}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load PubMedQA data
    print("📥 Loading PubMedQA data...")
    loader = PubMedQALoader()
    all_questions = loader.load(split="pqa_labeled")
    questions = all_questions[:num_documents]
    
    print(f"   Loaded {len(questions)} questions (from {len(all_questions)} total)")

    # Step 2: Extract documents from contexts
    print("📄 Extracting documents from contexts...")
    documents: list[CorpusDocument] = []
    seen_pmids: set[str] = set()
    
    for q in questions:
        # PubMedQA questions have context containing the abstract
        if hasattr(q, 'context') and q.context:
            # Context is typically a list of sentences or a dict
            if isinstance(q.context, dict):
                for pmid, text in q.context.items():
                    if pmid not in seen_pmids:
                        seen_pmids.add(pmid)
                        if isinstance(text, list):
                            text = " ".join(text)
                        documents.append(CorpusDocument(
                            pmid=pmid,
                            title=f"PubMedQA Document {pmid}",
                            abstract=text,
                        ))
            elif isinstance(q.context, list):
                # Use question_id as pseudo-PMID
                pmid = q.question_id
                if pmid not in seen_pmids:
                    seen_pmids.add(pmid)
                    text = " ".join(q.context) if isinstance(q.context[0], str) else str(q.context)
                    documents.append(CorpusDocument(
                        pmid=pmid,
                        title=f"PubMedQA Document {pmid}",
                        abstract=text,
                    ))
            elif isinstance(q.context, str):
                pmid = q.question_id
                if pmid not in seen_pmids:
                    seen_pmids.add(pmid)
                    documents.append(CorpusDocument(
                        pmid=pmid,
                        title=f"PubMedQA Document {pmid}",
                        abstract=q.context,
                    ))
    
    print(f"   Extracted {len(documents)} unique documents")

    # Step 3: Chunk documents
    print("✂️ Chunking documents...")
    chunker = RecursiveChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    all_chunks: list[Chunk] = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
    
    print(f"   Created {len(all_chunks)} chunks")

    # Step 4: Create embedder
    print(f"🔢 Creating {embeddings_provider} embedder...")
    if embeddings_provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("❌ Error: OPENAI_API_KEY not set")
            print("   Set it in .env or export OPENAI_API_KEY=...")
            sys.exit(1)
        embedder = OpenAIEmbedder(model=embeddings_model)
    else:
        embedder = LocalEmbedder(model=embeddings_model)

    # Step 5: Create FAISS store and add chunks
    print("📊 Building FAISS index...")
    store = FAISSStore(embedder=embedder, metric="cosine")
    store.add_chunks(all_chunks, show_progress=True)
    
    print(f"   Index size: {store._index.ntotal} vectors")

    # Step 6: Save index
    print(f"💾 Saving index to {output_dir}...")
    store.save(output_dir)

    # Step 7: Save config
    config = {
        "num_documents": len(documents),
        "num_chunks": len(all_chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embeddings_provider": embeddings_provider,
        "embeddings_model": embeddings_model,
        "index_size": store._index.ntotal,
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print()
    print("✅ Demo index built successfully!")
    print(f"   Documents: {len(documents)}")
    print(f"   Chunks: {len(all_chunks)}")
    print(f"   Index vectors: {store._index.ntotal}")
    print(f"   Output: {output_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build a pre-optimized FAISS index for the demo"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(PROJECT_ROOT / "demo" / "index"),
        help="Output directory for the index (default: demo/index)",
    )
    parser.add_argument(
        "--num-documents",
        "-n",
        type=int,
        default=500,
        help="Number of documents to include (default: 500)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=350,
        help="Chunk size (default: 350)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=40,
        help="Chunk overlap (default: 40)",
    )
    parser.add_argument(
        "--embeddings",
        choices=["openai", "local"],
        default="openai",
        help="Embeddings provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-large",
        help="Embeddings model (default: text-embedding-3-large)",
    )
    
    args = parser.parse_args()
    
    build_demo_index(
        output_dir=Path(args.output),
        num_documents=args.num_documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embeddings_provider=args.embeddings,
        embeddings_model=args.model,
    )


if __name__ == "__main__":
    main()

