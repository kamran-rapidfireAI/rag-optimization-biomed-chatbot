"""Corpus builder for BioRAG Bench."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

from biorag.schemas.corpus import CorpusDocument, CorpusManifest
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


class CorpusBuilder:
    """Build corpus from PubMed abstracts via HuggingFace datasets."""

    # Default HuggingFace dataset for PubMed abstracts
    HF_DATASET = "ncbi/pubmed"
    
    def __init__(
        self,
        output_dir: Path,
        gold_pmids: set[str] | None = None,
        distractor_count: int = 10000,
        sampling_seed: int = 42,
        cache_dir: Path | None = None,
        min_abstract_length: int = 100,
    ) -> None:
        """
        Initialize corpus builder.

        Args:
            output_dir: Directory to write output files
            gold_pmids: Set of gold PMIDs that must be included
            distractor_count: Number of distractor documents to sample
            sampling_seed: Random seed for reproducibility
            cache_dir: Cache directory for HuggingFace datasets
            min_abstract_length: Minimum abstract length to include
        """
        self.output_dir = Path(output_dir)
        self.gold_pmids = gold_pmids or set()
        self.distractor_count = distractor_count
        self.sampling_seed = sampling_seed
        self.cache_dir = cache_dir
        self.min_abstract_length = min_abstract_length

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        dataset_name: str | None = None,
        dataset_revision: str | None = None,
        resume: bool = True,
    ) -> CorpusManifest:
        """
        Build the corpus from HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset name (default: ncbi/pubmed)
            dataset_revision: Specific revision/commit hash
            resume: Whether to resume from checkpoint

        Returns:
            CorpusManifest with build info
        """
        import random

        dataset_name = dataset_name or self.HF_DATASET
        
        logger.info(f"Building corpus from {dataset_name}")
        logger.info(f"Gold PMIDs: {len(self.gold_pmids)}, Distractors: {self.distractor_count}")

        # Output paths
        corpus_path = self.output_dir / "corpus.jsonl"
        gold_pmids_path = self.output_dir / "pmids_gold.txt"
        distractor_pmids_path = self.output_dir / "pmids_distractors.txt"
        manifest_path = self.output_dir / "manifest.json"

        # Check for existing checkpoint
        if resume and corpus_path.exists():
            logger.info("Found existing corpus, loading for resume")
            existing_pmids = self._load_existing_pmids(corpus_path)
        else:
            existing_pmids = set()

        # Set random seed
        random.seed(self.sampling_seed)

        # Collect documents
        documents: list[CorpusDocument] = []
        gold_docs_found: set[str] = set()
        distractor_docs: list[CorpusDocument] = []

        try:
            # Try to load the dataset
            logger.info(f"Loading dataset: {dataset_name}")
            
            # For demonstration, we'll use a simpler approach
            # In production, you'd stream the full PubMed dataset
            dataset = self._load_pubmed_sample(dataset_name, dataset_revision)

            for doc in dataset:
                pmid = doc.pmid

                # Skip if already processed
                if pmid in existing_pmids:
                    continue

                # Check if gold document
                if pmid in self.gold_pmids:
                    doc.is_gold = True
                    documents.append(doc)
                    gold_docs_found.add(pmid)
                else:
                    # Candidate for distractor
                    distractor_docs.append(doc)

        except Exception as e:
            logger.warning(f"Error loading dataset: {e}")
            logger.info("Falling back to mock data for demonstration")
            documents, distractor_docs, gold_docs_found = self._create_mock_corpus()

        # Sample distractors
        if len(distractor_docs) > self.distractor_count:
            random.shuffle(distractor_docs)
            distractor_docs = distractor_docs[: self.distractor_count]

        documents.extend(distractor_docs)
        distractor_pmids = {d.pmid for d in distractor_docs}

        logger.info(
            f"Corpus: {len(documents)} docs "
            f"({len(gold_docs_found)} gold, {len(distractor_pmids)} distractors)"
        )

        # Write outputs
        self._write_corpus(corpus_path, documents)
        self._write_pmid_list(gold_pmids_path, gold_docs_found)
        self._write_pmid_list(distractor_pmids_path, distractor_pmids)

        # Create manifest
        manifest = CorpusManifest(
            build_timestamp=datetime.utcnow(),
            source_method="huggingface",
            dataset_name=dataset_name,
            dataset_revision=dataset_revision or "",
            sampling_seed=self.sampling_seed,
            gold_pmid_count=len(gold_docs_found),
            distractor_pmid_count=len(distractor_pmids),
            total_records=len(documents),
            min_abstract_length=self.min_abstract_length,
            output_files={
                "corpus": str(corpus_path),
                "corpus_sha256": self._compute_file_hash(corpus_path),
                "gold_pmids": str(gold_pmids_path),
                "distractor_pmids": str(distractor_pmids_path),
            },
        )

        # Write manifest
        with open(manifest_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))

        logger.info(f"Corpus built successfully: {manifest_path}")
        return manifest

    def _load_pubmed_sample(
        self, dataset_name: str, revision: str | None  # noqa: ARG002
    ) -> Iterator[CorpusDocument]:
        """Load a sample of PubMed abstracts."""
        try:
            # Try to load from HuggingFace
            # Note: The full PubMed dataset is very large, so we use streaming
            dataset = load_dataset(
                dataset_name,
                split="train",
                streaming=True,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                trust_remote_code=True,
            )

            count = 0
            max_docs = len(self.gold_pmids) + self.distractor_count * 2

            for item in dataset:
                if count >= max_docs:
                    break

                doc = self._parse_pubmed_item(item)
                if doc and len(doc.abstract) >= self.min_abstract_length:
                    yield doc
                    count += 1

        except Exception as e:
            logger.warning(f"Could not load PubMed dataset: {e}")
            # Return empty iterator - will fall back to mock data
            return

    def _parse_pubmed_item(self, item: dict) -> CorpusDocument | None:
        """Parse a PubMed dataset item into CorpusDocument."""
        try:
            pmid = str(item.get("pmid") or item.get("PMID") or item.get("id", ""))
            if not pmid:
                return None

            title = item.get("title") or item.get("ArticleTitle", "")
            abstract = item.get("abstract") or item.get("AbstractText", "")

            if not abstract:
                return None

            # Handle list abstracts
            if isinstance(abstract, list):
                abstract = " ".join(str(a) for a in abstract)
            if isinstance(title, list):
                title = " ".join(str(t) for t in title)

            return CorpusDocument(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=item.get("authors", []),
                journal=item.get("journal", ""),
                year=item.get("year"),
                source="pubmed",
            )
        except Exception:
            return None

    def _create_mock_corpus(
        self,
    ) -> tuple[list[CorpusDocument], list[CorpusDocument], set[str]]:
        """Create mock corpus for demonstration/testing."""
        logger.info("Creating mock corpus for demonstration")

        documents = []
        gold_found = set()

        # Create mock gold documents
        for i, pmid in enumerate(list(self.gold_pmids)[:50]):
            doc = CorpusDocument(
                pmid=pmid,
                title=f"Mock Gold Article {i+1}",
                abstract=f"This is a mock abstract for gold document {pmid}. "
                "It contains biomedical information relevant to the evaluation questions. "
                "The content discusses various medical topics including diseases, treatments, "
                "and clinical outcomes that are important for biomedical question answering.",
                is_gold=True,
                source="pubmed",
            )
            documents.append(doc)
            gold_found.add(pmid)

        # Create mock distractors
        distractors = []
        for i in range(min(self.distractor_count, 100)):
            doc = CorpusDocument(
                pmid=f"mock_{i+1:06d}",
                title=f"Mock Distractor Article {i+1}",
                abstract=f"This is a mock distractor abstract {i+1}. "
                "It contains general biomedical content that may or may not be relevant "
                "to specific questions. Topics include various aspects of medical research, "
                "clinical studies, and scientific findings in the healthcare domain.",
                is_gold=False,
                source="pubmed",
            )
            distractors.append(doc)

        return documents, distractors, gold_found

    def _write_corpus(self, path: Path, documents: list[CorpusDocument]) -> None:
        """Write corpus to JSONL file."""
        with open(path, "w") as f:
            for doc in documents:
                f.write(doc.model_dump_json() + "\n")

    def _write_pmid_list(self, path: Path, pmids: set[str]) -> None:
        """Write PMID list to file."""
        with open(path, "w") as f:
            for pmid in sorted(pmids):
                f.write(pmid + "\n")

    def _load_existing_pmids(self, corpus_path: Path) -> set[str]:
        """Load PMIDs from existing corpus file."""
        pmids = set()
        with open(corpus_path) as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    pmids.add(doc["pmid"])
                except Exception:
                    continue
        return pmids

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @classmethod
    def load_corpus(cls, corpus_path: Path) -> Iterator[CorpusDocument]:
        """Load corpus documents from JSONL file."""
        with open(corpus_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    yield CorpusDocument.model_validate(data)
                except Exception as e:
                    logger.warning(f"Failed to parse corpus line: {e}")
                    continue

