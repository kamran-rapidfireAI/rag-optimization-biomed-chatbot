"""PubMedQA dataset loader for BioRAG Bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Literal

from datasets import load_dataset

from biorag.schemas.evaluation import PubMedQAQuestion
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


class PubMedQALoader:
    """Loader for PubMedQA dataset from HuggingFace or local files."""

    # HuggingFace dataset info
    HF_DATASET = "qiaojin/PubMedQA"
    HF_CONFIG = "pqa_labeled"  # Use labeled subset

    def __init__(
        self,
        source: str = "huggingface",
        local_path: Path | None = None,
        cache_dir: Path | None = None,
        config: str | None = None,
    ) -> None:
        """
        Initialize PubMedQA loader.

        Args:
            source: Data source - "huggingface" or "local"
            local_path: Path to local JSON file if source is "local"
            cache_dir: Cache directory for HuggingFace datasets
            config: HuggingFace dataset config (pqa_labeled, pqa_unlabeled, pqa_artificial)
        """
        self.source = source
        self.local_path = local_path
        self.cache_dir = cache_dir
        self.config = config or self.HF_CONFIG

    def load(self, split: str = "train") -> list[PubMedQAQuestion]:
        """
        Load PubMedQA questions from the specified split.

        Args:
            split: Dataset split to load ("train", "validation", "test")

        Returns:
            List of PubMedQAQuestion objects
        """
        if self.source == "huggingface":
            return self._load_from_huggingface(split)
        elif self.source == "local":
            return self._load_from_local()
        else:
            raise ValueError(f"Unknown source: {self.source}")

    def _load_from_huggingface(self, split: str) -> list[PubMedQAQuestion]:
        """Load from HuggingFace datasets."""
        logger.info(f"Loading PubMedQA from HuggingFace ({self.config}, {split} split)")

        try:
            dataset = load_dataset(
                self.HF_DATASET,
                self.config,
                split=split,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                trust_remote_code=True,
            )
        except ValueError as e:
            # Some configs might not have all splits
            logger.warning(f"Split '{split}' not available: {e}")
            # Try to load available split
            if split != "train":
                return self._load_from_huggingface("train")
            raise

        questions = []
        for idx, item in enumerate(dataset):
            try:
                question = self._parse_hf_item(item, idx, split)
                if question:
                    questions.append(question)
            except Exception as e:
                logger.warning(f"Failed to parse item {idx}: {e}")
                continue

        logger.info(f"Loaded {len(questions)} PubMedQA questions")
        return questions

    def _parse_hf_item(
        self, item: dict, idx: int, split: str
    ) -> PubMedQAQuestion | None:
        """Parse a HuggingFace dataset item into PubMedQAQuestion."""
        # Extract fields
        pmid = str(item.get("pubid") or item.get("pmid") or item.get("id") or idx)
        question_id = f"pubmedqa_{pmid}"

        question_text = item.get("question", "")
        if not question_text:
            return None

        # Context can be a list or dict
        context = item.get("context", {})
        if isinstance(context, dict):
            # Handle nested context format
            context_list = context.get("contexts", [])
            if not context_list:
                # Try to extract from other fields
                context_list = []
                if "BACKGROUND" in context:
                    context_list.append(context["BACKGROUND"])
                if "METHODS" in context:
                    context_list.append(context["METHODS"])
                if "RESULTS" in context:
                    context_list.append(context["RESULTS"])
                if "CONCLUSIONS" in context:
                    context_list.append(context["CONCLUSIONS"])
        elif isinstance(context, list):
            context_list = context
        else:
            context_list = [str(context)] if context else []

        # Long answer
        long_answer = item.get("long_answer", "")
        if isinstance(long_answer, list):
            long_answer = " ".join(long_answer)

        # Label
        label = item.get("final_decision") or item.get("label", "maybe")
        if isinstance(label, str):
            label = label.lower()
        if label not in ("yes", "no", "maybe"):
            label = "maybe"

        return PubMedQAQuestion(
            question_id=question_id,
            question_text=question_text,
            context=context_list,
            long_answer=long_answer,
            label=label,
            pmid=pmid,
            split=split if split in ("train", "dev", "test") else "test",
        )

    def _load_from_local(self) -> list[PubMedQAQuestion]:
        """Load from local JSON file."""
        if not self.local_path or not self.local_path.exists():
            raise FileNotFoundError(f"Local file not found: {self.local_path}")

        logger.info(f"Loading PubMedQA from local file: {self.local_path}")

        with open(self.local_path) as f:
            data = json.load(f)

        # Handle dict format (keyed by PMID)
        if isinstance(data, dict):
            items = [(k, v) for k, v in data.items()]
        else:
            items = [(str(i), v) for i, v in enumerate(data)]

        questions = []
        for pmid, item in items:
            try:
                item["pmid"] = pmid
                question = self._parse_hf_item(item, 0, "test")
                if question:
                    questions.append(question)
            except Exception as e:
                logger.warning(f"Failed to parse item {pmid}: {e}")
                continue

        logger.info(f"Loaded {len(questions)} PubMedQA questions from local file")
        return questions

    def iter_questions(self, split: str = "train") -> Iterator[PubMedQAQuestion]:
        """Iterate over questions without loading all into memory."""
        questions = self.load(split)
        yield from questions

    def get_pmids(self, split: str = "train") -> set[str]:
        """Get all PMIDs from the dataset."""
        questions = self.load(split)
        pmids = {q.pmid for q in questions}
        logger.info(f"Found {len(pmids)} unique PMIDs in PubMedQA {split}")
        return pmids

    def get_label_distribution(
        self, split: str = "train"
    ) -> dict[Literal["yes", "no", "maybe"], int]:
        """Get the distribution of labels in the dataset."""
        questions = self.load(split)
        distribution: dict[Literal["yes", "no", "maybe"], int] = {
            "yes": 0,
            "no": 0,
            "maybe": 0,
        }
        for q in questions:
            if q.label in distribution:
                distribution[q.label] += 1
        return distribution

    def sample_questions(
        self,
        n: int,
        split: str = "train",
        seed: int = 42,
        stratify_by_label: bool = True,
    ) -> list[PubMedQAQuestion]:
        """
        Sample n questions from the dataset.

        Args:
            n: Number of questions to sample
            split: Dataset split
            seed: Random seed for reproducibility
            stratify_by_label: Whether to stratify by label

        Returns:
            List of sampled questions
        """
        import random

        questions = self.load(split)
        random.seed(seed)

        if not stratify_by_label:
            return random.sample(questions, min(n, len(questions)))

        # Stratify by label
        by_label: dict[str, list[PubMedQAQuestion]] = {}
        for q in questions:
            by_label.setdefault(q.label, []).append(q)

        # Sample proportionally
        sampled = []
        per_label = n // len(by_label)
        remainder = n % len(by_label)

        for i, (_label, qs) in enumerate(sorted(by_label.items())):
            count = per_label + (1 if i < remainder else 0)
            sampled.extend(random.sample(qs, min(count, len(qs))))

        random.shuffle(sampled)
        return sampled[:n]

