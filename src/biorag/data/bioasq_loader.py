"""BioASQ dataset loader for BioRAG Bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

from biorag.schemas.evaluation import BioASQQuestion
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


class BioASQLoader:
    """Loader for BioASQ dataset from HuggingFace or local files."""

    # HuggingFace dataset info
    HF_DATASET = "bigbio/bioasq_task_b"
    HF_CONFIG = "bioasq_task_b_source"

    def __init__(
        self,
        source: str = "huggingface",
        local_path: Path | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        """
        Initialize BioASQ loader.

        Args:
            source: Data source - "huggingface" or "local"
            local_path: Path to local JSON file if source is "local"
            cache_dir: Cache directory for HuggingFace datasets
        """
        self.source = source
        self.local_path = local_path
        self.cache_dir = cache_dir
        self._dataset = None

    def load(self, split: str = "train") -> list[BioASQQuestion]:
        """
        Load BioASQ questions from the specified split.

        Args:
            split: Dataset split to load ("train", "validation", "test")

        Returns:
            List of BioASQQuestion objects
        """
        if self.source == "huggingface":
            return self._load_from_huggingface(split)
        elif self.source == "local":
            return self._load_from_local()
        else:
            raise ValueError(f"Unknown source: {self.source}")

    def _load_from_huggingface(self, split: str) -> list[BioASQQuestion]:
        """Load from HuggingFace datasets."""
        logger.info(f"Loading BioASQ from HuggingFace ({split} split)")

        try:
            dataset = load_dataset(
                self.HF_DATASET,
                self.HF_CONFIG,
                split=split,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Failed to load with config, trying without: {e}")
            # Try without config
            dataset = load_dataset(
                self.HF_DATASET,
                split=split,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                trust_remote_code=True,
            )

        questions = []
        for idx, item in enumerate(dataset):
            try:
                question = self._parse_hf_item(item, idx)
                if question:
                    questions.append(question)
            except Exception as e:
                logger.warning(f"Failed to parse item {idx}: {e}")
                continue

        logger.info(f"Loaded {len(questions)} BioASQ questions")
        return questions

    def _parse_hf_item(self, item: dict, idx: int) -> BioASQQuestion | None:
        """Parse a HuggingFace dataset item into BioASQQuestion."""
        # Handle different schema versions
        question_id = item.get("id") or item.get("question_id") or f"bioasq_{idx}"
        question_text = item.get("body") or item.get("question") or item.get("text", "")

        if not question_text:
            return None

        # Determine question type
        qtype = item.get("type", "factoid").lower()
        if qtype not in ("yesno", "factoid", "list", "summary"):
            qtype = "factoid"  # Default

        # Extract gold documents
        gold_pmids = []
        documents = item.get("documents", [])
        if isinstance(documents, list):
            for doc in documents:
                if isinstance(doc, str):
                    # Extract PMID from URL like "http://www.ncbi.nlm.nih.gov/pubmed/12345"
                    pmid = doc.split("/")[-1] if "/" in doc else doc
                    gold_pmids.append(pmid)
                elif isinstance(doc, dict):
                    pmid = doc.get("pmid") or doc.get("id", "")
                    if pmid:
                        gold_pmids.append(str(pmid))

        # Extract snippets
        gold_snippets = []
        snippets = item.get("snippets", [])
        if isinstance(snippets, list):
            for snippet in snippets:
                if isinstance(snippet, str):
                    gold_snippets.append(snippet)
                elif isinstance(snippet, dict):
                    text = snippet.get("text", "")
                    if text:
                        gold_snippets.append(text)

        # Extract answers
        exact_answer = None
        ideal_answer = None

        if qtype == "yesno":
            exact_answer = item.get("exact_answer", "")
            if isinstance(exact_answer, list):
                exact_answer = exact_answer[0] if exact_answer else ""
        elif qtype == "factoid":
            exact_answer = item.get("exact_answer", [])
            if isinstance(exact_answer, str):
                exact_answer = exact_answer
            elif isinstance(exact_answer, list) and exact_answer:
                # Flatten nested lists
                if isinstance(exact_answer[0], list):
                    exact_answer = [a for sublist in exact_answer for a in sublist]
                exact_answer = exact_answer[0] if len(exact_answer) == 1 else exact_answer
        elif qtype == "list":
            exact_answer = item.get("exact_answer", [])
            # Flatten nested lists
            if isinstance(exact_answer, list) and exact_answer and isinstance(exact_answer[0], list):
                exact_answer = [a for sublist in exact_answer for a in sublist]
        elif qtype == "summary":
            ideal_answer = item.get("ideal_answer", "")
            if isinstance(ideal_answer, list):
                ideal_answer = ideal_answer[0] if ideal_answer else ""

        return BioASQQuestion(
            question_id=str(question_id),
            question_text=question_text,
            question_type=qtype,
            gold_pmids=gold_pmids,
            gold_snippets=gold_snippets,
            exact_answer=exact_answer,
            ideal_answer=ideal_answer,
        )

    def _load_from_local(self) -> list[BioASQQuestion]:
        """Load from local JSON file."""
        if not self.local_path or not self.local_path.exists():
            raise FileNotFoundError(f"Local file not found: {self.local_path}")

        logger.info(f"Loading BioASQ from local file: {self.local_path}")

        with open(self.local_path) as f:
            data = json.load(f)

        # Handle official BioASQ format
        if "questions" in data:
            data = data["questions"]

        questions = []
        for idx, item in enumerate(data):
            try:
                question = self._parse_hf_item(item, idx)
                if question:
                    questions.append(question)
            except Exception as e:
                logger.warning(f"Failed to parse item {idx}: {e}")
                continue

        logger.info(f"Loaded {len(questions)} BioASQ questions from local file")
        return questions

    def iter_questions(self, split: str = "train") -> Iterator[BioASQQuestion]:
        """Iterate over questions without loading all into memory."""
        questions = self.load(split)
        yield from questions

    def get_gold_pmids(self, split: str = "train") -> set[str]:
        """Get all gold PMIDs from the dataset."""
        questions = self.load(split)
        pmids = set()
        for q in questions:
            pmids.update(q.gold_pmids)
        logger.info(f"Found {len(pmids)} unique gold PMIDs in BioASQ {split}")
        return pmids

    def sample_questions(
        self,
        n: int,
        split: str = "train",
        seed: int = 42,
        stratify_by_type: bool = True,
    ) -> list[BioASQQuestion]:
        """
        Sample n questions from the dataset.

        Args:
            n: Number of questions to sample
            split: Dataset split
            seed: Random seed for reproducibility
            stratify_by_type: Whether to stratify by question type

        Returns:
            List of sampled questions
        """
        import random

        questions = self.load(split)
        random.seed(seed)

        if not stratify_by_type:
            return random.sample(questions, min(n, len(questions)))

        # Stratify by question type
        by_type: dict[str, list[BioASQQuestion]] = {}
        for q in questions:
            by_type.setdefault(q.question_type, []).append(q)

        # Sample proportionally
        sampled = []
        per_type = n // len(by_type)
        remainder = n % len(by_type)

        for i, (_qtype, qs) in enumerate(sorted(by_type.items())):
            count = per_type + (1 if i < remainder else 0)
            sampled.extend(random.sample(qs, min(count, len(qs))))

        random.shuffle(sampled)
        return sampled[:n]

