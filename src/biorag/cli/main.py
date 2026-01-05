"""BioRAG Bench CLI - Command Line Interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from biorag import __version__
from biorag.schemas.config import load_config
from biorag.utils.logging import setup_logging

# Create Typer app
app = typer.Typer(
    name="biorag",
    help="BioRAG Bench - Biomedical RAG Optimization Pipeline",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"BioRAG Bench v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """BioRAG Bench - Biomedical RAG Optimization Pipeline."""
    pass


@app.command()
def info(
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
    ),
) -> None:
    """Show current configuration and system info."""
    config = load_config(config_path)
    setup_logging(level=config.logging.level, json_format=config.logging.json_format)

    console.print("[bold blue]BioRAG Bench Configuration[/bold blue]")
    console.print(f"  Version: {__version__}")
    console.print(f"  LLM: {config.llm.provider}/{config.llm.model}")
    console.print(f"  Embeddings: {config.embeddings.provider}/{config.embeddings.model}")
    console.print(f"  Chunking: {config.chunking.type} ({config.chunking.chunk_size} tokens)")
    console.print(f"  Retrieval: {config.retrieval.mode} (k={config.retrieval.k})")
    console.print(f"  Rerank: {'enabled' if config.rerank.enabled else 'disabled'}")


@app.command()
def ingest_bioasq(
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    split: str = typer.Option("train", "--split", "-s", help="Dataset split"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
) -> None:
    """Load and validate BioASQ dataset."""
    from biorag.data.bioasq_loader import BioASQLoader

    config = load_config(config_path)
    setup_logging(level=config.logging.level, json_format=config.logging.json_format)

    console.print(f"[bold blue]Loading BioASQ dataset ({split} split)...[/bold blue]")

    loader = BioASQLoader(
        source="huggingface",
        cache_dir=config.paths.cache_dir,
    )

    try:
        questions = loader.load(split=split)
        console.print(f"[green]✓ Loaded {len(questions)} questions[/green]")

        # Show type distribution
        types: dict[str, int] = {}
        for q in questions:
            types[q.question_type] = types.get(q.question_type, 0) + 1

        console.print("\n[bold]Question type distribution:[/bold]")
        for qtype, count in sorted(types.items()):
            console.print(f"  {qtype}: {count}")

        # Get gold PMIDs
        pmids = loader.get_gold_pmids(split=split)
        console.print(f"\n[bold]Gold PMIDs:[/bold] {len(pmids)}")

        # Save to output if specified
        if output_dir:
            import json

            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"bioasq_{split}.jsonl"
            with open(output_path, "w") as f:
                for q in questions:
                    f.write(q.model_dump_json() + "\n")
            console.print(f"[green]✓ Saved to {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error loading BioASQ: {e}[/red]")
        raise typer.Exit(code=1) from None


@app.command()
def ingest_pubmedqa(
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    split: str = typer.Option("train", "--split", "-s", help="Dataset split"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
) -> None:
    """Load and validate PubMedQA dataset."""
    from biorag.data.pubmedqa_loader import PubMedQALoader

    config = load_config(config_path)
    setup_logging(level=config.logging.level, json_format=config.logging.json_format)

    console.print(f"[bold blue]Loading PubMedQA dataset ({split} split)...[/bold blue]")

    loader = PubMedQALoader(
        source="huggingface",
        cache_dir=config.paths.cache_dir,
    )

    try:
        questions = loader.load(split=split)
        console.print(f"[green]✓ Loaded {len(questions)} questions[/green]")

        # Show label distribution
        dist = loader.get_label_distribution(split=split)
        console.print("\n[bold]Label distribution:[/bold]")
        for label, count in sorted(dist.items()):
            console.print(f"  {label}: {count}")

        # Get PMIDs
        pmids = loader.get_pmids(split=split)
        console.print(f"\n[bold]Unique PMIDs:[/bold] {len(pmids)}")

        # Save to output if specified
        if output_dir:
            import json

            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"pubmedqa_{split}.jsonl"
            with open(output_path, "w") as f:
                for q in questions:
                    f.write(q.model_dump_json() + "\n")
            console.print(f"[green]✓ Saved to {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error loading PubMedQA: {e}[/red]")
        raise typer.Exit(code=1) from None


@app.command()
def build_corpus(
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    distractor_count: int = typer.Option(10000, "--distractors", "-d"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Build corpus from PubMed abstracts."""
    from biorag.data.bioasq_loader import BioASQLoader
    from biorag.data.corpus_builder import CorpusBuilder
    from biorag.data.pubmedqa_loader import PubMedQALoader

    config = load_config(config_path)
    setup_logging(level=config.logging.level, json_format=config.logging.json_format)

    output_path = output_dir or config.paths.data_dir / "processed" / "corpus"
    output_path.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]Building corpus...[/bold blue]")

    # Collect gold PMIDs from both datasets
    console.print("Collecting gold PMIDs from BioASQ...")
    bioasq_loader = BioASQLoader(source="huggingface", cache_dir=config.paths.cache_dir)
    gold_pmids = bioasq_loader.get_gold_pmids()

    console.print("Collecting PMIDs from PubMedQA...")
    pubmedqa_loader = PubMedQALoader(source="huggingface", cache_dir=config.paths.cache_dir)
    gold_pmids.update(pubmedqa_loader.get_pmids())

    console.print(f"[green]Total gold PMIDs: {len(gold_pmids)}[/green]")

    # Build corpus
    builder = CorpusBuilder(
        output_dir=output_path,
        gold_pmids=gold_pmids,
        distractor_count=distractor_count,
        sampling_seed=seed,
        cache_dir=config.paths.cache_dir,
    )

    manifest = builder.build()
    console.print(f"[green]✓ Corpus built: {manifest.total_records} documents[/green]")
    console.print(f"  Gold: {manifest.gold_pmid_count}")
    console.print(f"  Distractors: {manifest.distractor_pmid_count}")


@app.command()
def index_faiss(
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    corpus_path: Path | None = typer.Option(None, "--corpus", help="Path to corpus.jsonl"),
) -> None:
    """Build FAISS index from corpus."""
    console.print("[yellow]Command will be implemented in Phase 2[/yellow]")


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Query to retrieve documents for"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    k: int = typer.Option(10, "--k", help="Number of results to return"),
) -> None:
    """Retrieve chunks for a query (debugging)."""
    console.print("[yellow]Command will be implemented in Phase 3[/yellow]")


@app.command()
def eval(
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    dataset: str = typer.Option("bioasq", "--dataset", "-d", help="Dataset: bioasq or pubmedqa"),
    split: str = typer.Option("train", "--split", "-s", help="Dataset split"),
    max_questions: int | None = typer.Option(None, "--max", "-m", help="Max questions to evaluate"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    index_path: Path | None = typer.Option(None, "--index", "-i", help="Path to FAISS index"),
    run_id: str | None = typer.Option(None, "--run-id", help="Run identifier"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick eval with 10 samples"),
    seed: int = typer.Option(42, "--seed", help="Random seed for sampling"),
) -> None:
    """Run evaluation on BioASQ or PubMedQA golden suite."""
    from biorag.eval.harness import EvalProgress, EvaluationHarness
    from biorag.indexing.faiss_store import FAISSStore
    from biorag.pipeline.rag import RAGPipeline

    config = load_config(config_path)
    setup_logging(level=config.logging.level, json_format=config.logging.json_format)

    console.print(f"[bold blue]Running evaluation on {dataset}[/bold blue]")

    # Validate dataset
    if dataset not in ("bioasq", "pubmedqa"):
        console.print(f"[red]Unknown dataset: {dataset}. Use 'bioasq' or 'pubmedqa'[/red]")
        raise typer.Exit(code=1)

    # Set up output directory
    out_dir = output_dir or config.paths.runs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Progress callback
    def show_progress(progress: EvalProgress) -> None:
        console.print(
            f"  Progress: {progress.completed}/{progress.total} "
            f"({progress.progress_pct:.1f}%) - "
            f"Elapsed: {progress.elapsed_seconds:.1f}s",
            end="\r",
        )

    try:
        # Create pipeline
        pipeline = RAGPipeline(config=config)

        # Load FAISS index if provided
        if index_path:
            console.print(f"Loading FAISS index from {index_path}...")
            pipeline.load_index(index_path)

        # Create harness
        harness = EvaluationHarness(
            pipeline=pipeline,
            config=config,
            output_dir=out_dir,
        )

        # Quick mode
        if quick:
            console.print(f"[yellow]Quick mode: evaluating 10 samples[/yellow]")
            result = harness.quick_eval(
                dataset=dataset,  # type: ignore
                num_questions=10,
                split=split,
                seed=seed,
            )
        else:
            # Load questions
            console.print(f"Loading {dataset} questions ({split} split)...")
            questions = harness.load_golden_suite(
                dataset=dataset,  # type: ignore
                split=split,
                max_questions=max_questions,
                seed=seed,
            )
            console.print(f"[green]✓ Loaded {len(questions)} questions[/green]")

            # Run evaluation
            console.print("Running evaluation...")
            if dataset == "bioasq":
                result = harness.evaluate_bioasq(
                    questions,  # type: ignore
                    run_id=run_id,
                    progress_callback=show_progress,
                )
            else:
                result = harness.evaluate_pubmedqa(
                    questions,  # type: ignore
                    run_id=run_id,
                    progress_callback=show_progress,
                )

        console.print()  # New line after progress
        console.print("[green]✓ Evaluation complete![/green]")

        # Display results
        if result.metrics:
            console.print("\n[bold]Results:[/bold]")
            console.print(f"  Run ID: {result.run_id}")
            console.print(f"  Questions: {result.metrics.num_questions}")
            console.print(f"  Abstained: {result.metrics.num_abstained}")

            console.print("\n[bold]Retrieval Metrics:[/bold]")
            for name, metric in result.metrics.retrieval_metrics.items():
                if metric.count > 0:
                    console.print(f"  {name}: {metric.value:.4f}")

            console.print("\n[bold]Answer Metrics:[/bold]")
            for name, metric in result.metrics.answer_metrics.items():
                if metric.count > 0 and name not in result.metrics.retrieval_metrics:
                    console.print(f"  {name}: {metric.value:.4f}")

            console.print("\n[bold]Latency (avg):[/bold]")
            console.print(f"  Retrieval: {result.metrics.avg_retrieval_latency_ms:.1f}ms")
            console.print(f"  Rerank: {result.metrics.avg_rerank_latency_ms:.1f}ms")
            console.print(f"  Generation: {result.metrics.avg_generation_latency_ms:.1f}ms")
            console.print(f"  Total: {result.metrics.avg_total_latency_ms:.1f}ms")

            console.print("\n[bold]Cost:[/bold]")
            console.print(f"  Input tokens: {result.metrics.total_input_tokens:,}")
            console.print(f"  Output tokens: {result.metrics.total_output_tokens:,}")
            console.print(f"  Estimated cost: ${result.metrics.estimated_cost_usd:.4f}")
            console.print(f"  Cache hit rate: {result.metrics.cache_hit_rate:.1%}")

        if not quick:
            console.print(f"\n[green]Results saved to {out_dir / result.run_id}[/green]")

    except Exception as e:
        console.print(f"[red]Error during evaluation: {e}[/red]")
        raise typer.Exit(code=1) from None


@app.command()
def sweep(
    sweep_config_path: Path = typer.Argument(..., help="Path to sweep config YAML"),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Base config path"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    index_path: Path | None = typer.Option(None, "--index", "-i", help="Path to FAISS index"),
    parallel: bool = typer.Option(False, "--parallel", "-p", help="Use RapidFire AI parallel execution"),
    num_shards: int = typer.Option(4, "--shards", help="Number of shards for RapidFire AI"),
    no_rapidfire: bool = typer.Option(False, "--no-rapidfire", help="Disable RapidFire AI"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show configs without running"),
) -> None:
    """Run hyperparameter sweep using RapidFire AI for parallel execution."""
    from biorag.experiments.sweep import RAPIDFIRE_AVAILABLE, SweepRunner, generate_grid
    from biorag.schemas.experiments import SweepConfig

    config = load_config(config_path)
    setup_logging(level=config.logging.level, json_format=config.logging.json_format)

    console.print(f"[bold blue]Loading sweep config: {sweep_config_path}[/bold blue]")

    # Show RapidFire AI status
    if RAPIDFIRE_AVAILABLE and not no_rapidfire:
        console.print("[green]✓ RapidFire AI available - hyperparallelized execution enabled[/green]")
    elif no_rapidfire:
        console.print("[yellow]RapidFire AI disabled by --no-rapidfire flag[/yellow]")
    else:
        console.print("[yellow]RapidFire AI not available - using sequential execution[/yellow]")
        console.print("  Install with: pip install rapidfireai")

    try:
        # Load sweep config
        sweep_cfg = SweepConfig.from_yaml(str(sweep_config_path))

        # Override parallel settings from CLI
        if parallel:
            sweep_cfg.parallel = True

        # Override output dir
        if output_dir:
            sweep_cfg.output_dir = str(output_dir)

        # Generate grid and show info
        configs = generate_grid(sweep_cfg.parameters)
        console.print(f"\n[bold]Sweep: {sweep_cfg.name}[/bold]")
        console.print(f"  Description: {sweep_cfg.description}")
        console.print(f"  Total configurations: {len(configs)}")
        console.print(f"  Dataset: {sweep_cfg.dataset}")
        console.print(f"  Max questions per run: {sweep_cfg.max_questions or 'all'}")
        console.print(f"  Parallel: {sweep_cfg.parallel}")

        console.print("\n[bold]Parameters:[/bold]")
        for param in sweep_cfg.parameters:
            values = param.get_values()
            console.print(f"  {param.path}: {len(values)} values")
            if len(values) <= 5:
                console.print(f"    → {values}")
            else:
                console.print(f"    → {values[:3]} ... {values[-2:]}")

        if dry_run:
            console.print("\n[yellow]Dry run mode - showing first 5 configurations:[/yellow]")
            for i, cfg in enumerate(configs[:5]):
                console.print(f"\n  Config {i + 1}:")
                for path, value in _flatten_dict(cfg):
                    console.print(f"    {path}: {value}")
            if len(configs) > 5:
                console.print(f"\n  ... and {len(configs) - 5} more configurations")
            return

        # Confirm before running
        console.print()
        if not typer.confirm(f"Run {len(configs)} configurations?"):
            console.print("[yellow]Sweep cancelled[/yellow]")
            raise typer.Exit(code=0)

        # Progress callback
        def show_progress(current: int, total: int, result: object) -> None:
            if result:
                from biorag.experiments.runner import RunResult
                r = result if isinstance(result, RunResult) else None
                if r:
                    status = "[green]✓[/green]" if r.status == "completed" else "[red]✗[/red]"
                    console.print(
                        f"  {status} [{current}/{total}] {r.run_id}: "
                        f"metric={r.primary_metric:.4f}, time={r.latency.total_ms:.0f}ms"
                    )
                else:
                    console.print(f"  [{current}/{total}] Completed")
            else:
                console.print(f"  [{current}/{total}] Running...")

        # Create runner and execute sweep
        runner = SweepRunner(
            base_config=config,
            output_dir=output_dir or config.paths.runs_dir,
            index_path=index_path,
            use_rapidfire=not no_rapidfire,
        )

        console.print("\n[bold]Running sweep...[/bold]")
        result = runner.run_sweep(
            sweep_cfg,
            progress_callback=show_progress,
            num_shards=num_shards,
        )

        # Display results
        console.print("\n[green]✓ Sweep complete![/green]")
        console.print(f"\n[bold]Results:[/bold]")
        console.print(f"  Total runs: {result.total_runs}")
        console.print(f"  Completed: {result.completed_runs}")
        console.print(f"  Failed: {result.failed_runs}")
        console.print(f"  Total cost: ${result.total_cost_usd:.4f}")

        if result.best_run_id:
            console.print(f"\n[bold]Best Configuration:[/bold]")
            console.print(f"  Run ID: {result.best_run_id}")
            console.print(f"  Primary metric: {result.best_metric:.4f}")
            console.print(f"  Average metric: {result.average_metric:.4f}")

        console.print(f"\n[green]Leaderboard saved to: {result.leaderboard_path}[/green]")

    except FileNotFoundError as e:
        console.print(f"[red]Error: Sweep config not found: {e}[/red]")
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]Error running sweep: {e}[/red]")
        raise typer.Exit(code=1) from None


def _flatten_dict(d: dict, parent_key: str = "") -> list[tuple[str, object]]:
    """Flatten nested dict into list of (path, value) tuples."""
    items: list[tuple[str, object]] = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key))
        else:
            items.append((new_key, v))
    return items


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", "-r"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    index_path: Path | None = typer.Option(None, "--index", "-i", help="Path to FAISS index"),
    log_level: str = typer.Option("info", "--log-level", "-l"),
) -> None:
    """Start the FastAPI server for the BioRAG Bench API."""
    from biorag.api.app import run_server

    config = load_config(config_path)
    setup_logging(level=config.logging.level, json_format=config.logging.json_format)

    console.print(f"[bold blue]Starting BioRAG Bench API Server[/bold blue]")
    console.print(f"  Host: {host}")
    console.print(f"  Port: {port}")
    console.print(f"  Reload: {reload}")
    console.print(f"  Config: {config_path or 'default'}")
    console.print(f"  Index: {index_path or 'default'}")
    console.print()
    console.print(f"[green]API docs available at http://{host}:{port}/docs[/green]")
    console.print()

    run_server(
        host=host,
        port=port,
        reload=reload,
        config_path=config_path,
        index_path=index_path,
        log_level=log_level,
    )


if __name__ == "__main__":
    app()
