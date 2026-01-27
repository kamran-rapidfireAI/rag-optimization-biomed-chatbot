# Accelerating RAG Pipeline Optimization with RapidFire AI

> **TL;DR:** We used RapidFire AI to optimize a biomedical RAG pipeline, improving accuracy from 42% to 54% (+28% relative) in under 40 minutes — testing 16 configurations in parallel for just $0.04.

---

## The Hidden Cost of RAG Experimentation

Building a production-quality Retrieval-Augmented Generation (RAG) pipeline is deceptively complex. The architecture looks simple on paper: retrieve relevant documents, optionally rerank them, then generate an answer grounded in evidence. But beneath this simplicity lies a combinatorial explosion of tunable parameters:

- **Chunking**: How do you split documents? What chunk size? How much overlap?
- **Retrieval**: Similarity search or MMR? How many documents to fetch? What score threshold?
- **Reranking**: Which cross-encoder model? How many candidates to rerank?
- **Prompting**: Which template produces the best citation behavior?

Each parameter interacts with the others in non-obvious ways. A chunk size that works well with similarity search might underperform with MMR. A reranking model that excels on one dataset might struggle on another.

Traditional experimentation approaches tackle this problem sequentially: run one configuration, wait for results, adjust, repeat. This creates a brutal bottleneck. With API rate limits, GPU constraints, and the sheer volume of evaluations needed, teams often spend **days or weeks** exploring a fraction of the parameter space.

**There's a better way.**

---

## Introducing RapidFire AI: The Hyperparallelization Engine

[RapidFire AI](https://github.com/RapidFireAI/rapidfireai) transforms slow, sequential AI experimentation into rapid, intelligent workflows. Instead of running one configuration at a time, RapidFire AI orchestrates **parallel execution across your entire parameter grid**, while intelligently managing resources and rate limits.

Key capabilities:

- **16-24x throughput improvement** over sequential runs
- **Shard-based scheduling** — compare many configurations concurrently, even on a single GPU
- **Intelligent rate limit management** — automatic handling of API constraints
- **Real-time control** — stop, resume, or clone-modify experiments on the fly

But rather than enumerate features, let's see it in action.

---

## Case Study: BioRAG Bench

We built **BioRAG Bench** — a biomedical question-answering system that retrieves evidence from 100,000+ PubMed abstracts to answer medical research questions. The system needed to:

1. Retrieve relevant evidence from a large corpus
2. Generate grounded answers with verifiable citations
3. Abstain when evidence is insufficient (critical for medical applications)
4. Achieve high accuracy on established benchmarks (PubMedQA, BioASQ)

### The Architecture

```
Question → Embedding → FAISS Retrieval → Cross-Encoder Reranking → LLM Generation → Answer with Citations
```

### The Optimization Challenge

We had a baseline configuration that achieved **42% accuracy** on PubMedQA. Good, but not good enough. We needed to find the optimal combination across:

| Parameter | Options |
|-----------|---------|
| Retrieval mode | `similarity`, `mmr` |
| Documents to retrieve (k) | 5, 10, 15 |
| Fetch candidates (fetch_k) | 20, 50 |
| Reranking enabled | `true`, `false` |
| Rerank final_k | 5, 8 |
| Prompt template | v1, v2 |

That's potentially **hundreds of configurations** to evaluate. Doing this sequentially would take days.

---

## RapidFire AI in Action

We built a sweep runner that integrates with RapidFire AI for hyperparallelized execution. Our parameter sweep is defined in a simple YAML configuration:

```yaml
name: demo_optimization
description: "Find optimal config for Baseline vs Optimized comparison"

parameters:
  - path: retrieval.mode
    range:
      type: choice
      values: [similarity, mmr]

  - path: retrieval.k
    range:
      type: grid
      values: [5, 10, 15]

  - path: rerank.enabled
    range:
      type: choice
      values: [true, false]

  - path: rerank.final_k
    range:
      type: grid
      values: [5, 8]

# Parallel execution settings
parallel: true
max_parallel: 4
```

Then launched the sweep:

```bash
biorag sweep configs/sweeps/demo_optimization.yaml --parallel
```

### What Happened Next

Our sweep runner, powered by RapidFire AI's hyperparallelization engine:

1. **Generated the configuration grid** — 16 distinct configurations from the parameter space
2. **Executed in parallel** — 4 runs at a time, with RapidFire AI managing GPU and API rate limits
3. **Aggregated results** — automatic leaderboard generation
4. **Produced artifacts** — per-run metrics, predictions, and configs for reproducibility

**Time elapsed: ~40 minutes**

**Total cost: $0.04 USD** (thanks to intelligent caching and batch optimization)

---

## The Results

```
┌─────────────────────────────────────────────────────────┐
│                 OPTIMIZATION RESULTS                    │
├──────────────────┬──────────────────┬──────────────────┤
│   🔵 Baseline    │  🟢 Optimized    │  📈 Improvement  │
│       42%        │       54%        │      +28%        │
│    Accuracy      │    Accuracy      │    Relative      │
└──────────────────┴──────────────────┴──────────────────┘
```

### What Changed?

| Setting | Baseline | Optimized |
|---------|----------|-----------|
| Retrieval Mode | `similarity` | `mmr` (Maximal Marginal Relevance) |
| Documents Retrieved | 5 | 15 |
| Fetch Candidates | 20 | 50 |
| Evidence Diversity | Low | High (MMR balances relevance + diversity) |

The winning insight: **more diverse evidence leads to better answers**. The MMR algorithm ensures the LLM sees varied perspectives from the literature, reducing the risk of answer bias from redundant sources.

### Sweep Statistics

| Metric | Value |
|--------|-------|
| Configurations tested | 16 |
| Parallel workers | 4 |
| Completion rate | 100% (0 failures) |
| Total sweep cost | $0.04 USD |
| Best accuracy found | 54% |
| Worst accuracy found | 36% |

### Per-Query Improvements

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Evidence chunks retrieved | 5 | 15 (3x more context) |
| Answer diversity | Limited | High (MMR diversity) |
| Abstention rate | 12% | 12% (maintained safety) |

---

## The Leaderboard

RapidFire AI automatically generates a ranked leaderboard of all configurations:

```csv
rank,run_id,accuracy,retrieval_mode,k,rerank_enabled
1,run_bcc5d2,48%,similarity,5,false
2,run_987b74,48%,similarity,5,false
3,run_cd0561,44%,similarity,10,false
4,run_c8263d,44%,similarity,10,false
5,run_1794b3,36%,similarity,5,true
...
```

Every run is fully reproducible — configuration, predictions, and metrics are preserved as artifacts.

---

## Why RapidFire AI?

### Before: Sequential Experimentation

```
Config 1 → wait 10 min → Config 2 → wait 10 min → ... → Config 16
Total time: ~160 minutes (2.5+ hours)
```

### After: Hyperparallelized with RapidFire AI

```
                    ┌─ Config 1 ─┐
                    ├─ Config 2 ─┤
RapidFire AI ───────┼─ Config 3 ─┼───► Leaderboard
                    ├─ Config 4 ─┤
                    └─ Config N ─┘
Total time: ~40 minutes (4x faster)
```

### Key Benefits

1. **Speed**: Explore your entire parameter space in hours, not days
2. **Cost efficiency**: Intelligent caching prevents redundant API calls
3. **Reproducibility**: Every run produces artifacts for full experiment tracking
4. **Control**: Stop, resume, or modify experiments in real-time
5. **Scale**: Works on a single GPU or distributed across a cluster

---

## Getting Started with RapidFire AI

### 1. Install RapidFire AI

```bash
pip install rapidfireai
```

### 2. Integrate with Your Pipeline

RapidFire AI provides the hyperparallelization engine. You integrate it with your evaluation pipeline using its API:

```python
from rapidfireai import RFGridSearch, RFLangChainRagSpec

# Define your RAG spec following RapidFire AI patterns
spec = RFLangChainRagSpec(
    preprocess=your_preprocess_fn,
    postprocess=your_postprocess_fn,
    compute_metrics=your_metrics_fn,
    accumulate_metrics=your_accumulate_fn,
)

# Run hyperparallelized sweep
grid_search = RFGridSearch(spec, config_grid)
results = grid_search.run()
```

### 3. Define Your Parameter Space

Create configuration files for the parameters you want to sweep (the schema is up to you):

```yaml
parameters:
  - path: retrieval.k
    values: [5, 10, 20]
  - path: rerank.enabled
    values: [true, false]
```

### 4. Analyze Results

Your integration produces:
- `leaderboard.csv` — ranked configurations by your primary metric
- `sweep_summary.json` — aggregate statistics
- Per-run directories with configs, predictions, and metrics

---

## Conclusion: Stop Waiting, Start Optimizing

RAG pipeline optimization doesn't have to be a slow, frustrating process. With RapidFire AI, you can:

- **Explore more** — test hundreds of configurations where you previously tested dozens
- **Move faster** — reduce experimentation time from days to hours
- **Stay confident** — every result is reproducible with full artifact tracking

Our BioRAG Bench journey took us from 42% to 54% accuracy — a **28% relative improvement** — in under 40 minutes and for less than 5 cents in API costs.

**Your RAG pipeline has untapped potential. RapidFire AI helps you find it.**

---

## Resources

- [RapidFire AI GitHub](https://github.com/RapidFireAI/rapidfireai)
- [BioRAG Bench Repository](https://github.com/your-org/rag-optimization-biomed-chatbot)
- [BioRAG Bench Demo on HuggingFace Spaces](https://huggingface.co/spaces/yourusername/biorag-bench)

---

## Technical Appendix: BioRAG Bench Statistics

### System Overview

| Component | Technology |
|-----------|------------|
| Vector Store | FAISS (Flat index, 1.2 GB) |
| Embeddings | OpenAI text-embedding-3-large (dim: 3072) |
| LLM | GPT-4o-mini |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Corpus Size | ~100,000 PubMed abstract chunks |

### Benchmark Performance

| Dataset | Baseline | Optimized |
|---------|----------|-----------|
| PubMedQA | 42% | 54% |
| Improvement | — | +28% relative |

### Latency Breakdown (per query)

| Stage | Time |
|-------|------|
| FAISS Retrieval | ~1.5s |
| Reranking (GPU) | ~20ms |
| LLM Generation | ~4.5s |
| **Total** | **~6s** |

### Cost Analysis

| Operation | Cost |
|-----------|------|
| 100-question evaluation | ~$0.05 |
| 16-config sweep | ~$0.04 |
| Per-query (avg) | ~$0.0005 |

---

*Built with RapidFire AI, LangChain, FAISS, and OpenAI. For research and educational purposes only.*

