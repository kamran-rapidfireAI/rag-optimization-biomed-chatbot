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
    dataset: str = typer.Option("bioasq", "--dataset", "-d"),
) -> None:
    """Run evaluation on golden suite."""
    console.print("[yellow]Command will be implemented in Phase 6[/yellow]")


@app.command()
def sweep(
    sweep_config: Path = typer.Argument(..., help="Path to sweep config YAML"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Run hyperparameter sweep."""
    console.print("[yellow]Command will be implemented in Phase 7[/yellow]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", "-r"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Start the FastAPI server."""
    console.print("[yellow]Command will be implemented in Phase 5[/yellow]")


if __name__ == "__main__":
    app()
