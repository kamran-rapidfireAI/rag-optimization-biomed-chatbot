"""Evaluation harness for BioRAG Bench.

Orchestrates evaluation runs over golden suites with:
- Batch evaluation for efficiency
- Full reproducibility tracking
- Cost and latency reporting
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from biorag.eval.bioasq_eval import BioASQEvaluator, BioASQMetrics
from biorag.eval.pubmedqa_eval import PubMedQAEvaluator, PubMedQAMetrics
from biorag.pipeline.rag import RAGPipeline, RAGResult
from biorag.schemas.config import BioRAGConfig, load_config
from biorag.schemas.evaluation import (
    BioASQQuestion,
    EvalPrediction,
    EvalResult,
    MetricResult,
    PubMedQAQuestion,
    RetrievalResult,
    RunMetrics,
)
from biorag.utils.caching import LLMCache
from biorag.utils.cost import CostTracker
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvalProgress:
    """Progress tracking for evaluation run."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    abstained: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def progress_pct(self) -> float:
        return (self.completed / self.total * 100) if self.total > 0 else 0.0

    @property
    def estimated_remaining(self) -> float:
        if self.completed == 0:
            return 0.0
        rate = self.elapsed_seconds / self.completed
        return rate * (self.total - self.completed)


def get_git_commit() -> str | None:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return None


class EvaluationHarness:
    """
    Evaluation harness for running and tracking RAG evaluations.

    Features:
    - Batch evaluation with progress tracking
    - Full reproducibility info (git commit, config, model versions)
    - Cost and latency tracking
    - Support for BioASQ and PubMedQA benchmarks
    """

    def __init__(
        self,
        pipeline: RAGPipeline | None = None,
        config: BioRAGConfig | None = None,
        config_path: str | Path | None = None,
        cache: LLMCache | None = None,
        cost_tracker: CostTracker | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize evaluation harness.

        Args:
            pipeline: Pre-built RAG pipeline (optional)
            config: BioRAG configuration
            config_path: Path to config file
            cache: LLM cache for response caching
            cost_tracker: Cost tracker for budget monitoring
            output_dir: Directory for saving evaluation artifacts
        """
        # Load config
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            self.config = load_config()

        # Initialize components
        self.cache = cache or LLMCache(
            cache_dir=self.config.paths.cache_dir / "llm_cache"
        )
        self.cost_tracker = cost_tracker or CostTracker(
            max_usd=self.config.cost.max_usd,
            max_total_tokens=self.config.cost.max_total_tokens,
            on_budget_exceeded=self.config.cost.on_budget_exceeded,
        )

        # Pipeline will be lazily created if not provided
        self._pipeline = pipeline

        # Output directory
        self.output_dir = Path(output_dir) if output_dir else self.config.paths.runs_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluators
        self._bioasq_evaluator = BioASQEvaluator()
        self._pubmedqa_evaluator = PubMedQAEvaluator()

        logger.info("Evaluation harness initialized")

    @property
    def pipeline(self) -> RAGPipeline:
        """Get or create RAG pipeline."""
        if self._pipeline is None:
            self._pipeline = RAGPipeline(
                config=self.config,
                cache=self.cache,
                cost_tracker=self.cost_tracker,
            )
        return self._pipeline

    def evaluate_bioasq(
        self,
        questions: Sequence[BioASQQuestion],
        run_id: str | None = None,
        max_questions: int | None = None,
        progress_callback: Callable[[EvalProgress], None] | None = None,
        save_results: bool = True,
    ) -> EvalResult:
        """
        Evaluate on BioASQ questions.

        Args:
            questions: List of BioASQ questions
            run_id: Optional run identifier
            max_questions: Maximum questions to evaluate (for testing)
            progress_callback: Callback for progress updates
            save_results: Whether to save results to disk

        Returns:
            EvalResult with predictions and metrics
        """
        run_id = run_id or f"bioasq_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Limit questions if specified
        if max_questions:
            questions = list(questions)[:max_questions]

        logger.info(f"Starting BioASQ evaluation: {len(questions)} questions")

        # Run predictions
        predictions = self._run_predictions(
            questions=[q.question_text for q in questions],
            question_ids=[q.question_id for q in questions],
            question_types=[q.question_type for q in questions],
            progress_callback=progress_callback,
        )

        # Evaluate
        metrics = self._bioasq_evaluator.evaluate(questions, predictions)

        # Build result
        result = self._build_eval_result(
            run_id=run_id,
            dataset="bioasq",
            predictions=predictions,
            retrieval_metrics=metrics.to_metric_results(),
            answer_metrics=metrics.to_metric_results(),
            num_questions=len(questions),
            num_abstained=metrics.num_abstained,
        )

        # Save if requested
        if save_results:
            self._save_results(run_id, result, metrics)

        return result

    def evaluate_pubmedqa(
        self,
        questions: Sequence[PubMedQAQuestion],
        run_id: str | None = None,
        max_questions: int | None = None,
        progress_callback: Callable[[EvalProgress], None] | None = None,
        save_results: bool = True,
    ) -> EvalResult:
        """
        Evaluate on PubMedQA questions.

        Args:
            questions: List of PubMedQA questions
            run_id: Optional run identifier
            max_questions: Maximum questions to evaluate (for testing)
            progress_callback: Callback for progress updates
            save_results: Whether to save results to disk

        Returns:
            EvalResult with predictions and metrics
        """
        run_id = run_id or f"pubmedqa_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Limit questions if specified
        if max_questions:
            questions = list(questions)[:max_questions]

        logger.info(f"Starting PubMedQA evaluation: {len(questions)} questions")

        # Run predictions
        predictions = self._run_predictions(
            questions=[q.question_text for q in questions],
            question_ids=[q.question_id for q in questions],
            question_types=["yesno"] * len(questions),  # PubMedQA is all yes/no/maybe
            progress_callback=progress_callback,
        )

        # Evaluate
        metrics = self._pubmedqa_evaluator.evaluate(questions, predictions)

        # Build result
        result = self._build_eval_result(
            run_id=run_id,
            dataset="pubmedqa",
            predictions=predictions,
            retrieval_metrics=metrics.to_metric_results(),
            answer_metrics=metrics.to_metric_results(),
            num_questions=len(questions),
            num_abstained=metrics.num_abstained,
        )

        # Save if requested
        if save_results:
            self._save_results(run_id, result, metrics)

        return result

    def _run_predictions(
        self,
        questions: Sequence[str],
        question_ids: Sequence[str],
        question_types: Sequence[str],
        progress_callback: Callable[[EvalProgress], None] | None = None,
    ) -> list[EvalPrediction]:
        """
        Run pipeline predictions for a list of questions.

        Args:
            questions: List of question texts
            question_ids: List of question IDs
            question_types: List of question types
            progress_callback: Progress callback

        Returns:
            List of EvalPrediction objects
        """
        predictions: list[EvalPrediction] = []
        progress = EvalProgress(total=len(questions))

        for question, qid, qtype in zip(questions, question_ids, question_types):
            try:
                # Check budget
                if self.cost_tracker.should_skip():
                    logger.warning(f"Budget exceeded, stopping at question {qid}")
                    break

                # Run pipeline
                result = self.pipeline.query(question, question_type=qtype)

                # Convert to EvalPrediction
                pred = self._result_to_prediction(qid, result)
                predictions.append(pred)

                if pred.abstained:
                    progress.abstained += 1

            except Exception as e:
                logger.error(f"Error processing question {qid}: {e}")
                progress.failed += 1

                # Create empty prediction for failed question
                predictions.append(
                    EvalPrediction(
                        question_id=qid,
                        predicted_answer="",
                        abstained=True,
                        abstention_reason=f"Error: {e}",
                    )
                )

            progress.completed += 1

            # Progress callback
            if progress_callback:
                progress_callback(progress)

            # Log progress periodically
            if progress.completed % 10 == 0 or progress.completed == len(questions):
                logger.info(
                    f"Progress: {progress.completed}/{progress.total} "
                    f"({progress.progress_pct:.1f}%), "
                    f"elapsed: {progress.elapsed_seconds:.1f}s"
                )

        return predictions

    def _result_to_prediction(
        self,
        question_id: str,
        result: RAGResult,
    ) -> EvalPrediction:
        """Convert RAGResult to EvalPrediction."""
        # Extract retrieved PMIDs
        retrieved_pmids = [c.pmid for c in result.reranked_chunks]

        # Convert chunks to RetrievalResult format
        retrieved_chunks = []
        for i, chunk in enumerate(result.reranked_chunks):
            retrieved_chunks.append(
                RetrievalResult(
                    pmid=chunk.pmid,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=chunk.score,
                    rank=i + 1,
                    rerank_score=chunk.rerank_score,
                    rerank_rank=chunk.rerank_rank,
                )
            )

        # Get token usage
        input_tokens = 0
        output_tokens = 0
        if result.generation_response:
            input_tokens = result.generation_response.input_tokens
            output_tokens = result.generation_response.output_tokens

        return EvalPrediction(
            question_id=question_id,
            retrieved_pmids=retrieved_pmids,
            retrieved_chunks=retrieved_chunks,
            predicted_answer=result.answer.answer,
            predicted_label=result.answer.label,
            abstained=result.answer.abstained,
            abstention_reason=result.answer.abstention_reason,
            retrieval_latency_ms=result.latency.retrieve_ms,
            rerank_latency_ms=result.latency.rerank_ms,
            generation_latency_ms=result.latency.generate_ms,
            total_latency_ms=result.latency.total_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_output=result.answer.model_dump(),
        )

    def _build_eval_result(
        self,
        run_id: str,
        dataset: str,
        predictions: list[EvalPrediction],
        retrieval_metrics: dict[str, MetricResult],
        answer_metrics: dict[str, MetricResult],
        num_questions: int,
        num_abstained: int,
    ) -> EvalResult:
        """Build complete EvalResult with all metadata."""
        # Calculate latency stats
        latencies = [p for p in predictions if p.total_latency_ms > 0]
        avg_retrieval = sum(p.retrieval_latency_ms for p in latencies) / len(latencies) if latencies else 0
        avg_rerank = sum(p.rerank_latency_ms for p in latencies) / len(latencies) if latencies else 0
        avg_generation = sum(p.generation_latency_ms for p in latencies) / len(latencies) if latencies else 0
        avg_total = sum(p.total_latency_ms for p in latencies) / len(latencies) if latencies else 0

        # Calculate token stats
        total_input = sum(p.input_tokens for p in predictions)
        total_output = sum(p.output_tokens for p in predictions)

        # Get cost stats
        cost_stats = self.cost_tracker.get_report()

        # Build run metrics
        run_metrics = RunMetrics(
            run_id=run_id,
            dataset=dataset,
            num_questions=num_questions,
            num_abstained=num_abstained,
            retrieval_metrics=retrieval_metrics,
            answer_metrics=answer_metrics,
            avg_retrieval_latency_ms=avg_retrieval,
            avg_rerank_latency_ms=avg_rerank,
            avg_generation_latency_ms=avg_generation,
            avg_total_latency_ms=avg_total,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            estimated_cost_usd=cost_stats.get("estimated_cost_usd", 0.0),
            cache_hit_rate=cost_stats.get("cache_hit_rate", 0.0),
        )

        # Build model versions
        model_versions = {
            "llm": f"{self.config.llm.provider}/{self.config.llm.model}",
            "embeddings": f"{self.config.embeddings.provider}/{self.config.embeddings.model}",
            "reranker": self.config.rerank.model if self.config.rerank.enabled else "none",
        }

        return EvalResult(
            run_id=run_id,
            config=self.config.model_dump(),
            git_commit=get_git_commit(),
            dataset_version=None,  # Could be set from data manifest
            model_versions=model_versions,
            random_seed=42,  # Default seed
            predictions=predictions,
            metrics=run_metrics,
        )

    def _save_results(
        self,
        run_id: str,
        result: EvalResult,
        metrics: BioASQMetrics | PubMedQAMetrics,
    ) -> None:
        """Save evaluation results to disk."""
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save full result
        with open(run_dir / "run.json", "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)

        # Save predictions separately for easier access
        with open(run_dir / "predictions.jsonl", "w") as f:
            for pred in result.predictions:
                f.write(json.dumps(pred.model_dump(), default=str) + "\n")

        # Save metrics summary
        if result.metrics:
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(result.metrics.model_dump(), f, indent=2, default=str)

        # Save config
        with open(run_dir / "config.json", "w") as f:
            json.dump(result.config, f, indent=2, default=str)

        logger.info(f"Results saved to {run_dir}")

    def load_golden_suite(
        self,
        dataset: Literal["bioasq", "pubmedqa"],
        split: str = "train",
        max_questions: int | None = None,
        seed: int = 42,
    ) -> list[BioASQQuestion] | list[PubMedQAQuestion]:
        """
        Load questions from a golden suite.

        Args:
            dataset: Dataset to load ("bioasq" or "pubmedqa")
            split: Dataset split
            max_questions: Maximum questions to load (samples if smaller than full)
            seed: Random seed for sampling

        Returns:
            List of questions
        """
        if dataset == "bioasq":
            from biorag.data.bioasq_loader import BioASQLoader

            loader = BioASQLoader(
                source="huggingface",
                cache_dir=self.config.paths.cache_dir,
            )
            if max_questions:
                return loader.sample_questions(max_questions, split=split, seed=seed)
            return loader.load(split)

        elif dataset == "pubmedqa":
            from biorag.data.pubmedqa_loader import PubMedQALoader

            loader = PubMedQALoader(
                source="huggingface",
                cache_dir=self.config.paths.cache_dir,
            )
            if max_questions:
                return loader.sample_questions(max_questions, split=split, seed=seed)
            return loader.load(split)

        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def quick_eval(
        self,
        dataset: Literal["bioasq", "pubmedqa"],
        num_questions: int = 10,
        split: str = "train",
        seed: int = 42,
    ) -> EvalResult:
        """
        Quick evaluation on a small sample for testing.

        Args:
            dataset: Dataset to evaluate on
            num_questions: Number of questions to sample
            split: Dataset split
            seed: Random seed

        Returns:
            EvalResult
        """
        logger.info(f"Running quick eval: {dataset}, {num_questions} questions")

        questions = self.load_golden_suite(
            dataset=dataset,
            split=split,
            max_questions=num_questions,
            seed=seed,
        )

        if dataset == "bioasq":
            return self.evaluate_bioasq(
                questions,  # type: ignore
                save_results=False,
            )
        else:
            return self.evaluate_pubmedqa(
                questions,  # type: ignore
                save_results=False,
            )

