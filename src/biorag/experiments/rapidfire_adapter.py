"""RapidFire AI adapter for BioRAG pipeline integration.

This module provides adapters and utilities to integrate the BioRAG pipeline
with RapidFire AI's hyperparallel experimentation framework.

Based on RapidFire AI tutorial patterns:
- Uses RFLangChainRagSpec for RAG pipeline configuration
- Uses RFOpenAIAPIModelConfig with rag parameter
- Provides preprocess_fn, postprocess_fn, compute_metrics_fn, accumulate_metrics_fn callbacks
- Uses RFGridSearch for hyperparameter sweeps

Reference: https://github.com/RapidFireAI/rapidfireai/tree/main/tutorial_notebooks/rag-contexteng
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Literal

from biorag.schemas.config import BioRAGConfig, load_config
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


# Check if rapidfireai is available
try:
    from rapidfireai import Experiment
    from rapidfireai.evals.automl import (
        List as RFList,
        RFGridSearch,
        RFLangChainRagSpec,
        RFOpenAIAPIModelConfig,
        RFPromptManager,
    )

    RAPIDFIRE_AVAILABLE = True
except ImportError as e:
    RAPIDFIRE_AVAILABLE = False
    Experiment = None  # type: ignore
    RFGridSearch = None  # type: ignore
    RFOpenAIAPIModelConfig = None  # type: ignore
    RFLangChainRagSpec = None  # type: ignore
    RFList = None  # type: ignore
    RFPromptManager = None  # type: ignore
    logger.warning(
        f"rapidfireai not available. Install with: pip install rapidfireai. Error: {e}"
    )


def check_rapidfire_available() -> bool:
    """Check if RapidFire AI is available."""
    return RAPIDFIRE_AVAILABLE


class BioRAGRapidFireAdapter:
    """
    Adapts BioRAG pipeline configuration to RapidFire AI format.

    This adapter creates RFLangChainRagSpec, RFOpenAIAPIModelConfig, and
    the required callback functions for RapidFire AI evaluation sweeps.

    Based on the patterns shown in:
    - rf-tutorial-rag-fiqa.ipynb
    - rf-tutorial-scifact-full-evaluation.ipynb
    """

    def __init__(
        self,
        base_config: BioRAGConfig | None = None,
        use_gpu: bool = True,
    ) -> None:
        """
        Initialize the adapter.

        Args:
            base_config: Base BioRAG configuration to use as defaults
            use_gpu: Whether to use GPU for embeddings and reranking
        """
        if not RAPIDFIRE_AVAILABLE:
            raise ImportError(
                "rapidfireai not installed. Install with: pip install rapidfireai"
            )

        self.base_config = base_config or load_config()
        self.use_gpu = use_gpu
        self.device = "cuda:0" if use_gpu else "cpu"

    def create_rag_spec(
        self,
        chunk_sizes: list[int] | None = None,
        search_types: list[str] | None = None,
        top_k: int = 10,
        reranker_top_n: list[int] | None = None,
        corpus_path: str | Path | None = None,
    ) -> "RFLangChainRagSpec":
        """
        Create RapidFire AI RAG specification.

        This wraps BioRAG's LangChain-based components into RFLangChainRagSpec
        for use with RapidFire AI's hyperparallel execution.

        Args:
            chunk_sizes: List of chunk sizes to sweep (creates List for grid search)
            search_types: List of search types to sweep ["similarity", "mmr"]
            top_k: Number of documents to retrieve
            reranker_top_n: List of reranker top_n values to sweep
            corpus_path: Path to corpus data (JSONL format)

        Returns:
            RFLangChainRagSpec configured for the sweep
        """
        from langchain_community.document_loaders import DirectoryLoader, JSONLoader
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

        # Default values
        chunk_sizes = chunk_sizes or [self.base_config.chunking.chunk_size]
        search_types = search_types or ["similarity"]
        reranker_top_n = reranker_top_n or [self.base_config.retrieval.rerank_top_k]

        # Create text splitters for each chunk size
        if len(chunk_sizes) > 1:
            text_splitters = RFList([
                RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    encoding_name="gpt2",
                    chunk_size=size,
                    chunk_overlap=self.base_config.chunking.chunk_overlap,
                )
                for size in chunk_sizes
            ])
        else:
            text_splitters = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="gpt2",
                chunk_size=chunk_sizes[0],
                chunk_overlap=self.base_config.chunking.chunk_overlap,
            )

        # Create search type list
        search_type = RFList(search_types) if len(search_types) > 1 else search_types[0]

        # Create reranker kwargs with top_n sweep
        reranker_kwargs = {
            "model_name": self.base_config.retrieval.reranker_model,
            "model_kwargs": {"device": self.device},
        }
        if len(reranker_top_n) > 1:
            reranker_kwargs["top_n"] = RFList(reranker_top_n)
        else:
            reranker_kwargs["top_n"] = reranker_top_n[0]

        # Build the RFLangChainRagSpec
        # Note: If corpus_path is not provided, we skip document_loader
        # and expect the RAG to be pre-indexed
        spec_kwargs = {
            "text_splitter": text_splitters,
            "embedding_cls": HuggingFaceEmbeddings,
            "embedding_kwargs": {
                "model_name": self.base_config.indexing.embedding_model,
                "model_kwargs": {"device": self.device},
                "encode_kwargs": {"normalize_embeddings": True, "batch_size": 128},
            },
            "vector_store": None,  # Uses FAISS by default
            "search_type": search_type,
            "search_kwargs": {"k": top_k},
            "reranker_cls": CrossEncoderReranker,
            "reranker_kwargs": reranker_kwargs,
            "enable_gpu_search": self.use_gpu,
        }

        if corpus_path:
            corpus_path = Path(corpus_path)
            spec_kwargs["document_loader"] = DirectoryLoader(
                path=str(corpus_path.parent),
                glob=corpus_path.name,
                loader_cls=JSONLoader,
                loader_kwargs={
                    "jq_schema": ".",
                    "content_key": "text",
                    "metadata_func": lambda record, metadata: {
                        "doc_id": record.get("id", record.get("_id", "")),
                        "title": record.get("title", ""),
                    },
                    "json_lines": True,
                    "text_content": False,
                },
                sample_seed=42,
            )

        return RFLangChainRagSpec(**spec_kwargs)

    def create_openai_config(
        self,
        rag_spec: "RFLangChainRagSpec",
        models: list[str] | None = None,
        temperatures: list[float] | None = None,
        max_tokens: int | None = None,
    ) -> "RFOpenAIAPIModelConfig" | "RFList":
        """
        Create RapidFire AI OpenAI model configuration.

        Args:
            rag_spec: The RFLangChainRagSpec to use for retrieval
            models: List of model names to sweep
            temperatures: List of temperatures to sweep
            max_tokens: Maximum tokens for generation

        Returns:
            RFOpenAIAPIModelConfig or RFList of configs for grid search
        """
        models = models or [self.base_config.llm.model]
        temperatures = temperatures or [self.base_config.llm.temperature]
        max_tokens = max_tokens or self.base_config.llm.max_tokens

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        configs = []
        for model in models:
            for temp in temperatures:
                config = RFOpenAIAPIModelConfig(
                    client_config={"api_key": api_key, "max_retries": 2},
                    model_config={
                        "model": model,
                        "temperature": temp,
                        "max_completion_tokens": max_tokens,
                    },
                    rpm_limit=10_000,
                    tpm_limit=2_000_000,
                    rag=rag_spec,
                    prompt_manager=None,
                )
                configs.append(config)

        if len(configs) == 1:
            return configs[0]
        return RFList(configs)

    def create_config_set(
        self,
        openai_configs: "RFOpenAIAPIModelConfig | RFList",
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """
        Create the complete config set for RFGridSearch.

        Args:
            openai_configs: OpenAI config(s) from create_openai_config
            batch_size: Batch size for processing

        Returns:
            Config set dict for RFGridSearch
        """
        return {
            "openai_config": openai_configs,
            "batch_size": batch_size,
            "preprocess_fn": self.create_preprocess_fn(),
            "postprocess_fn": self.create_postprocess_fn(),
            "compute_metrics_fn": self.create_compute_metrics_fn(),
            "accumulate_metrics_fn": self.create_accumulate_metrics_fn(),
            "online_strategy_kwargs": {
                "strategy_name": "normal",
                "confidence_level": 0.95,
                "use_fpc": True,
            },
        }

    def create_preprocess_fn(self):
        """
        Create the preprocess function for RapidFire AI.

        The preprocess function:
        1. Takes a batch of queries
        2. Uses rag.get_context() to retrieve relevant documents
        3. Serializes context and builds prompts
        """
        def preprocess_fn(
            batch: dict[str, list],
            rag: "RFLangChainRagSpec",
            prompt_manager: "RFPromptManager | None",
        ) -> dict[str, list]:
            """Prepare prompts with retrieved context for generation."""
            # Get context for all queries in the batch
            all_context = rag.get_context(batch_queries=batch["query"], serialize=False)

            # Extract retrieved document IDs
            retrieved_documents = [
                [doc.metadata.get("doc_id", str(i)) for i, doc in enumerate(docs)]
                for docs in all_context
            ]

            # Serialize context for prompts
            serialized_context = rag.serialize_documents(all_context)

            # Build conversational prompts
            system_prompt = (
                "You are a biomedical expert assistant. Answer questions accurately "
                "based on the provided scientific context. If the context doesn't contain "
                "enough information, say so clearly."
            )

            prompts = [
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
                    },
                ]
                for question, context in zip(batch["query"], serialized_context)
            ]

            return {
                "prompts": prompts,
                "retrieved_documents": retrieved_documents,
                **batch,
            }

        return preprocess_fn

    def create_postprocess_fn(self):
        """
        Create the postprocess function for RapidFire AI.

        The postprocess function processes model outputs and extracts answers.
        """
        def postprocess_fn(batch: dict[str, list]) -> dict[str, list]:
            """Process model outputs and prepare for metrics computation."""
            # Extract the generated text
            if "generated_text" in batch:
                # Clean up generated answers
                batch["answer"] = [
                    text.strip() for text in batch["generated_text"]
                ]
            return batch

        return postprocess_fn

    def create_compute_metrics_fn(self):
        """
        Create the compute_metrics function for RapidFire AI.

        Computes per-batch metrics including retrieval and generation quality.
        """
        def compute_ndcg_at_k(retrieved_docs: list, expected_docs: set, k: int = 5) -> float:
            """Compute NDCG@k for retrieval evaluation."""
            relevance = [1 if doc in expected_docs else 0 for doc in retrieved_docs[:k]]
            dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))

            ideal_length = min(k, len(expected_docs))
            ideal_relevance = [1] * ideal_length + [0] * (k - ideal_length)
            idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))

            return dcg / idcg if idcg > 0 else 0.0

        def compute_rr(retrieved_docs: list, expected_docs: set) -> float:
            """Compute Reciprocal Rank."""
            for i, doc in enumerate(retrieved_docs):
                if doc in expected_docs:
                    return 1 / (i + 1)
            return 0.0

        def compute_metrics_fn(batch: dict[str, list]) -> dict[str, dict[str, Any]]:
            """Compute metrics for a batch of predictions."""
            total_queries = len(batch["query"])

            # Initialize metric accumulators
            precisions, recalls, f1_scores = [], [], []
            ndcgs, rrs = [], []
            exact_matches = 0

            # Compute retrieval metrics if ground truth is available
            if "gold_pmids" in batch and "retrieved_documents" in batch:
                for retrieved, expected in zip(
                    batch["retrieved_documents"],
                    batch.get("gold_pmids", [[] for _ in range(total_queries)]),
                ):
                    expected_set = set(expected) if expected else set()
                    retrieved_set = set(retrieved)

                    if expected_set:
                        true_positives = len(expected_set.intersection(retrieved_set))
                        precision = true_positives / len(retrieved_set) if retrieved_set else 0
                        recall = true_positives / len(expected_set) if expected_set else 0
                        f1 = (
                            2 * precision * recall / (precision + recall)
                            if (precision + recall) > 0
                            else 0
                        )

                        precisions.append(precision)
                        recalls.append(recall)
                        f1_scores.append(f1)
                        ndcgs.append(compute_ndcg_at_k(list(retrieved), expected_set, k=5))
                        rrs.append(compute_rr(list(retrieved), expected_set))
                    else:
                        precisions.append(0)
                        recalls.append(0)
                        f1_scores.append(0)
                        ndcgs.append(0)
                        rrs.append(0)

            # Compute answer accuracy if expected answers are available
            if "expected" in batch and "answer" in batch:
                for answer, expected in zip(batch["answer"], batch["expected"]):
                    if answer and expected:
                        # Simple exact match (case-insensitive)
                        if str(answer).lower().strip() == str(expected).lower().strip():
                            exact_matches += 1

            accuracy = exact_matches / total_queries if total_queries > 0 else 0.0

            return {
                "Total": {"value": total_queries},
                "Accuracy": {"value": accuracy},
                "Precision": {"value": sum(precisions) / total_queries if precisions else 0.0},
                "Recall": {"value": sum(recalls) / total_queries if recalls else 0.0},
                "F1": {"value": sum(f1_scores) / total_queries if f1_scores else 0.0},
                "NDCG@5": {"value": sum(ndcgs) / total_queries if ndcgs else 0.0},
                "MRR": {"value": sum(rrs) / total_queries if rrs else 0.0},
            }

        return compute_metrics_fn

    def create_accumulate_metrics_fn(self):
        """
        Create the accumulate_metrics function for RapidFire AI.

        Aggregates metrics across all batches.
        """
        def accumulate_metrics_fn(
            aggregated_metrics: dict[str, list],
        ) -> dict[str, dict[str, Any]]:
            """Accumulate metrics across all batches."""
            num_queries_per_batch = [m["value"] for m in aggregated_metrics["Total"]]
            total_queries = sum(num_queries_per_batch)

            algebraic_metrics = ["Accuracy", "Precision", "Recall", "F1", "NDCG@5", "MRR"]

            result = {"Total": {"value": total_queries}}

            for metric in algebraic_metrics:
                if metric in aggregated_metrics:
                    weighted_sum = sum(
                        m["value"] * queries
                        for m, queries in zip(
                            aggregated_metrics[metric], num_queries_per_batch
                        )
                    )
                    result[metric] = {
                        "value": weighted_sum / total_queries if total_queries > 0 else 0.0,
                        "is_algebraic": True,
                        "value_range": (0, 1),
                    }

            return result

        return accumulate_metrics_fn


class BioRAGDatasetAdapter:
    """
    Adapter to convert BioRAG evaluation data to RapidFire AI dataset format.

    RapidFire AI expects a HuggingFace Dataset object with at least a 'query' column.
    """

    def __init__(self, config: BioRAGConfig | None = None) -> None:
        """
        Initialize the adapter.

        Args:
            config: BioRAG configuration
        """
        self.config = config or load_config()

    def load_bioasq_dataset(
        self,
        split: str = "train",
        max_questions: int | None = None,
        seed: int = 42,
    ) -> Any:
        """
        Load BioASQ questions as RapidFire AI dataset.

        Args:
            split: Dataset split to load
            max_questions: Maximum questions to load
            seed: Random seed for sampling

        Returns:
            HuggingFace Dataset for rapidfireai
        """
        from datasets import Dataset

        from biorag.data.bioasq_loader import BioASQLoader

        loader = BioASQLoader(
            source="huggingface",
            cache_dir=self.config.paths.cache_dir,
        )

        if max_questions:
            questions = loader.sample_questions(max_questions, split=split, seed=seed)
        else:
            questions = loader.load(split)

        # Convert to HuggingFace Dataset format
        data_dict = {
            "query": [q.question_text for q in questions],
            "question_id": [q.question_id for q in questions],
            "question_type": [q.question_type for q in questions],
            "expected": [q.exact_answer or q.ideal_answer for q in questions],
            "gold_pmids": [q.gold_pmids or [] for q in questions],
        }

        dataset = Dataset.from_dict(data_dict)
        if max_questions:
            dataset = dataset.shuffle(seed=seed).select(range(min(max_questions, len(dataset))))

        return dataset

    def load_pubmedqa_dataset(
        self,
        split: str = "train",
        max_questions: int | None = None,
        seed: int = 42,
    ) -> Any:
        """
        Load PubMedQA questions as RapidFire AI dataset.

        Args:
            split: Dataset split to load
            max_questions: Maximum questions to load
            seed: Random seed for sampling

        Returns:
            HuggingFace Dataset for rapidfireai
        """
        from datasets import Dataset

        from biorag.data.pubmedqa_loader import PubMedQALoader

        loader = PubMedQALoader(
            source="huggingface",
            cache_dir=self.config.paths.cache_dir,
        )

        if max_questions:
            questions = loader.sample_questions(max_questions, split=split, seed=seed)
        else:
            questions = loader.load(split)

        # Convert to HuggingFace Dataset format
        data_dict = {
            "query": [q.question_text for q in questions],
            "question_id": [q.question_id for q in questions],
            "expected": [q.label for q in questions],
            "pmid": [q.pmid for q in questions],
            "context": [q.context for q in questions],
        }

        dataset = Dataset.from_dict(data_dict)
        if max_questions:
            dataset = dataset.shuffle(seed=seed).select(range(min(max_questions, len(dataset))))

        return dataset


class RapidFireSweepRunner:
    """
    Run parameter sweeps using RapidFire AI's hyperparallel framework.

    This class wraps rapidfireai's Experiment class to run evaluation sweeps
    with shard-based scheduling and automatic optimization.

    Based on the patterns from:
    - rf-tutorial-rag-fiqa.ipynb
    - rf-tutorial-scifact-full-evaluation.ipynb
    """

    def __init__(
        self,
        experiment_name: str,
        base_config: BioRAGConfig | None = None,
        experiment_path: str | Path | None = None,
        use_gpu: bool = True,
    ) -> None:
        """
        Initialize the RapidFire sweep runner.

        Args:
            experiment_name: Name of the experiment
            base_config: Base BioRAG configuration
            experiment_path: Path to store experiment artifacts
            use_gpu: Whether to use GPU for embeddings and reranking
        """
        if not RAPIDFIRE_AVAILABLE:
            raise ImportError(
                "rapidfireai not installed. Install with: pip install rapidfireai"
            )

        self.experiment_name = experiment_name
        self.base_config = base_config or load_config()
        self.experiment_path = str(experiment_path) if experiment_path else None
        self.use_gpu = use_gpu

        # Create adapters
        self.rf_adapter = BioRAGRapidFireAdapter(self.base_config, use_gpu=use_gpu)
        self.dataset_adapter = BioRAGDatasetAdapter(self.base_config)

        logger.info(f"RapidFireSweepRunner initialized: {experiment_name}")

    def run_sweep(
        self,
        sweep_params: dict[str, list[Any]],
        dataset: Literal["bioasq", "pubmedqa"] = "bioasq",
        split: str = "train",
        max_questions: int | None = None,
        num_shards: int = 4,
        num_actors: int = 2,
        seed: int = 42,
        corpus_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Run a parameter sweep using RapidFire AI.

        Args:
            sweep_params: Dictionary mapping parameter names to lists of values
                Supported parameters:
                - "chunk_size": list of chunk sizes
                - "search_type": list of search types ["similarity", "mmr"]
                - "reranker_top_n": list of reranker top_n values
                - "model": list of model names
                - "temperature": list of temperatures
            dataset: Dataset to evaluate on ("bioasq" or "pubmedqa")
            split: Dataset split
            max_questions: Maximum questions per configuration
            num_shards: Number of shards for parallel processing
            num_actors: Number of Ray actors
            seed: Random seed
            corpus_path: Optional path to corpus for RAG

        Returns:
            Dictionary with run results
        """
        # Create experiment
        exp_kwargs = {"experiment_name": self.experiment_name, "mode": "evals"}
        if self.experiment_path:
            exp_kwargs["experiment_path"] = self.experiment_path

        experiment = Experiment(**exp_kwargs)

        try:
            # Build RFLangChainRagSpec with sweep parameters
            rag_spec = self.rf_adapter.create_rag_spec(
                chunk_sizes=sweep_params.get("chunk_size"),
                search_types=sweep_params.get("search_type"),
                reranker_top_n=sweep_params.get("reranker_top_n"),
                corpus_path=corpus_path,
            )

            # Build OpenAI config(s) with sweep parameters
            openai_configs = self.rf_adapter.create_openai_config(
                rag_spec=rag_spec,
                models=sweep_params.get("model"),
                temperatures=sweep_params.get("temperature"),
            )

            # Create the config set
            config_set = self.rf_adapter.create_config_set(
                openai_configs=openai_configs,
                batch_size=sweep_params.get("batch_size", [32])[0] if isinstance(
                    sweep_params.get("batch_size"), list
                ) else sweep_params.get("batch_size", 32),
            )

            # Create RFGridSearch
            config_group = RFGridSearch(config_set)

            # Load dataset
            if dataset == "bioasq":
                rf_dataset = self.dataset_adapter.load_bioasq_dataset(
                    split=split,
                    max_questions=max_questions,
                    seed=seed,
                )
            else:
                rf_dataset = self.dataset_adapter.load_pubmedqa_dataset(
                    split=split,
                    max_questions=max_questions,
                    seed=seed,
                )

            logger.info(
                f"Starting RapidFire sweep with {len(rf_dataset)} questions, "
                f"{num_shards} shards, {num_actors} actors"
            )

            # Run the sweep
            results = experiment.run_evals(
                config_group=config_group,
                dataset=rf_dataset,
                num_shards=num_shards,
                num_actors=num_actors,
                seed=seed,
            )

            logger.info(f"Sweep complete: {len(results)} configurations evaluated")
            return results

        finally:
            experiment.end()

    def get_results_dataframe(self, results: dict[str, Any]) -> Any:
        """
        Convert RapidFire results to a pandas DataFrame.

        Args:
            results: Results dict from run_sweep

        Returns:
            DataFrame with run results
        """
        import pandas as pd

        rows = []
        for run_id, (aggregated, cumulative) in results.items():
            row = {"run_id": run_id}
            for key, value in cumulative.items():
                if isinstance(value, dict) and "value" in value:
                    row[key] = value["value"]
                else:
                    row[key] = value
            rows.append(row)

        return pd.DataFrame(rows)


# Legacy compatibility
BioRAGConfigAdapter = BioRAGRapidFireAdapter
