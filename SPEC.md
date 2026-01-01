# BioRAG Bench — Requirements Specification

> **Elevator pitch:** Build a modular biomedical Retrieval-Augmented Generation (RAG) pipeline using **LangChain components** (FAISS vector store, configurable chunking, retriever settings, reranking, prompt variants) and evaluate it with **true golden tests** from **BioASQ Task B** and **PubMedQA** — **no manual labeling required**.

---

## Table of Contents

1. [Overview](#1-overview)
   - 1.1 [Constraints](#11-constraints)
   - 1.2 [Goals](#12-goals)
   - 1.3 [Scope (Question Types)](#13-scope-question-types)
2. [Data](#2-data)
   - 2.1 [BioASQ Task B](#21-bioasq-task-b)
   - 2.2 [PubMedQA](#22-pubmedqa)
   - 2.3 [Dataset Sources](#23-dataset-sources)
   - 2.4 [Golden Suite](#24-golden-suite)
3. [Functional Requirements](#3-functional-requirements)
   - 3.1 [FR-1] Data Loaders
   - 3.2 [FR-2] Corpus Builder
   - 3.3 [FR-3] Chunking
   - 3.4 [FR-4] Embedding
   - 3.5 [FR-5] Vector Store (FAISS)
   - 3.6 [FR-6] Retrieval
   - 3.7 [FR-7] Reranking
   - 3.8 [FR-8] Prompting & Generation
   - 3.9 [FR-9] Evaluation Harness
   - 3.10 [FR-10] Experiment Runner (RapidFire AI)
   - 3.11 [FR-11] API + CLI
4. [Non-Functional Requirements](#4-non-functional-requirements)
   - 4.1 [NFR-1] Reproducibility
   - 4.2 [NFR-2] Performance
   - 4.3 [NFR-3] Cost Control
   - 4.4 [NFR-4] Maintainability
   - 4.5 [NFR-5] Safety
5. [Technical Design](#5-technical-design)
   - 5.1 [Tech Stack](#51-tech-stack)
   - 5.2 [Architecture](#52-architecture)
   - 5.3 [Repo Layout](#53-repo-layout)
   - 5.4 [Config Design](#54-config-design)
   - 5.5 [GPU Usage](#55-gpu-usage)
6. [Deliverables](#6-deliverables)
   - 6.1 [Required](#61-required)
   - 6.2 [Valuable Add-ons](#62-valuable-add-ons)

---

## 1. Overview

### 1.1 Constraints

These constraints are baked into this specification:

| Constraint | Decision |
|------------|----------|
| **LLM** | Hosted API (OpenAI) for generation and embeddings |
| **Indexing** | FAISS only (no OpenSearch/Elastic) |
| **Timeline** | 2–4 weeks (portfolio-quality) |
| **Hardware** | Single local GPU (for cross-encoder reranking) |
| **Sweeps** | RapidFire AI for hyperparallelized optimization |
| **Benchmarks** | BioASQ Task B + PubMedQA |
| **Demo** | Gradio on HuggingFace Spaces |

**Optimization targets** (swept via RapidFire AI):

- Chunking strategy + parameters
- Retriever parameters (k, search type, MMR, score thresholds)
- Reranker parameters/models
- Prompt templates / citation formats / refusal policy

### 1.2 Goals

#### Primary goals

1. **End-to-end RAG pipeline**: ingest → chunk → embed → index (FAISS) → retrieve → rerank → generate → cite sources.
2. **Golden tests / automatic evaluation** using public labeled benchmarks:
   - BioASQ: gold relevant docs/snippets + gold answers (exact & ideal)
   - PubMedQA: gold label (yes/no/maybe) and supporting context
3. **Optimization-first workflow**:
   - Run parameter sweeps using RapidFire AI and track metrics + artifacts
   - Validate against a stable golden subset (e.g., 200 questions)

#### Secondary goals

- Demo web app (Gradio on HuggingFace Spaces) with side-by-side baseline vs optimized comparison
- Reproducible experiment tracking (run configs + metrics + artifacts)
- Fast iteration: caching embeddings, persistent FAISS index, batch evaluation

#### Non-goals

- Medical advice
- State-of-the-art leaderboard chasing (demonstrating engineering + measurement)
- Operating a large search cluster (kept intentionally simple)

### 1.3 Scope (Question Types)

This is an **evidence-grounded biomedical literature QA chatbot** (a "PubMed literature assistant"), not a general conversational assistant.

#### Best-fit question types (aligned with BioASQ + PubMedQA)

| Type | Description | Example |
|------|-------------|---------|
| **Yes/No/Maybe** | Strong fit; easy to score | "Does metformin reduce cancer risk in patients with diabetes?" |
| **Factoid** | Short, specific answers | "Which gene is mutated in Huntington's disease?" |
| **List** | Return a set of items with citations | "What are common adverse effects of amiodarone?" |
| **Summary** | Best for showing RAG value + citations | "Summarize evidence on SGLT2 inhibitors and heart failure outcomes." |

#### Out-of-scope queries (must refuse / reframe)

- Personalized medical advice: "Should I take X?" "What dose should I take?" "Is this safe for me?"
- Diagnosis / treatment planning for an individual

**Reframe pattern:** "I can't give personal medical advice, but I can summarize what published studies report about X for condition Y and cite the relevant PubMed articles."

---

## 2. Data

### 2.1 BioASQ Task B

Per question, you get:

- Gold relevant **documents** (PMIDs) and/or **snippets**
- Gold answers:
  - **Exact** answers (factoid/list/yes-no)
  - **Ideal** answers (summary paragraph)

These labels power both retrieval and generation evaluation.

**Created by:** BioASQ challenge organizers (a consortium-supported effort) with biomedical experts writing questions and providing gold documents/snippets and gold answers.

### 2.2 PubMedQA

Per sample, you get:

- Question (usually yes/no/maybe style)
- Label: **yes/no/maybe**
- Supporting abstract/context

This provides a second benchmark with clean auto-scoring.

**Created by:** Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W. Cohen, Xinghua Lu.

**Approach:** PubMedQA is treated as a **RAG + evidence task** that outputs yes/no/maybe with citations, consistent with BioASQ. The model retrieves evidence, generates an answer with citations, and the label is extracted for scoring.

### 2.3 Dataset Sources

> Record the **exact version/date** of each download in `data/raw/manifest.json`.

#### BioASQ Task B (official)

- BioASQ Participants Area → **Datasets** (Task B / "Task *b*" downloads)
- BioASQ Participants Area → Task *b* page (includes dataset JSON format notes)

#### PubMedQA (official)

- PubMedQA homepage (links to the dataset + code repository)
- PubMedQA GitHub repository (download instructions and splitting scripts)

#### Convenience mirrors (optional)

- Hugging Face datasets: `bigbio/bioasq_task_b` (community packaging)
- Hugging Face datasets: `qiaojin/PubMedQA` (community packaging)

#### PubMed abstracts for retrieval corpus

- **Chosen approach:** build the retrieval corpus from a **Hugging Face PubMed abstracts dataset**.
- Record the **dataset name**, **dataset version**, and **revision/commit hash** in `data/raw/manifest.json`.

#### Clarification: what "comes from PubMed"

This project uses **PubMed** in two different ways:

- **Underlying literature source (corpus):** the retrieval corpus is built from **PubMed records (titles/abstracts)**. PubMed is the original source of that text.
- **Benchmark datasets (labels/tests):** BioASQ and PubMedQA are **separate benchmark datasets** created by their respective organizers/authors. They *reference* PubMed articles (PMIDs) and are grounded in PubMed literature, but they are not "PubMed datasets" themselves.
- **Hugging Face role:** Hugging Face is the **distribution/packaging layer** used to download (a) the benchmark datasets and (b) a PubMed-abstracts corpus.

### 2.4 Golden Suite

Define a stable subset for consistent evaluation:

| Dataset | File | Size |
|---------|------|------|
| BioASQ | `bioasq_golden_200.jsonl` | 200 questions |
| PubMedQA | `pubmedqa_golden_500.jsonl` | 200–500 questions (depending on runtime) |

- Keep the subset deterministic (seeded sampling, committed to repo).
- Use this subset for:
  - Quick iteration during development
  - Comparing configs during sweeps
  - Manual regression checks before deploying new configs to demo

---

## 3. Functional Requirements

> **Convention:** Requirements are labeled with standard prefixes:
> - **[FR-#]** = Functional Requirement (what the system must *do*)
> - **[NFR-#]** = Non-Functional Requirement (how the system must *behave*) — see Section 4

### 3.1 [FR-1] Data Loaders

#### BioASQ loader

- Parse questions, types, gold docs/snippets, exact/ideal answers
- Assign stable `question_id` keys for caching and evaluation artifacts

#### PubMedQA loader

- Parse train/val/test splits
- Extract `question_id`, question, label, and supporting context/PMIDs (if available)

### 3.2 [FR-2] Corpus Builder (PubMed Abstracts)

Build a corpus of texts to retrieve from.

#### FR-2.1 Source of abstracts

**Option A — Hugging Face (selected for this project)**

- Use a Hugging Face dataset that contains PubMed abstracts at scale.
- Pin the **exact dataset revision** (commit hash) so the corpus is reproducible.
- Sample a **distractor set** from the same dataset using a fixed random seed.
- **Important practical note:** the popular `ncbi/pubmed` dataset represents the full PubMed baseline and is **very large** (tens of millions of records). For a 2–4 week project, prefer a smaller HF PubMed corpus.

**Option B — NCBI E-utilities (alternative)**

- Build a PMID list (all BioASQ gold PMIDs + all PubMedQA PMIDs + distractor PMIDs).
- Fetch abstracts via NCBI E-utilities with on-disk caching, retries, and rate limiting.

> **Decision:** This project uses **Option A (Hugging Face)**.

#### FR-2.2 Caching and reproducible downloads

- Cache at two layers:
  1. **HF dataset cache**: rely on the standard Hugging Face datasets cache (do not commit it). Record dataset **name + revision** so it can be rehydrated.
  2. **Materialized corpus cache**: write a deterministic, normalized `corpus.jsonl` under `data/processed/corpus/` (and a `chunks.jsonl` after chunking).
- Deterministic materialization:
  - Use a fixed `sampling_seed`
  - Record the exact split/shard selection logic
  - Persist the exact **PMID lists** used:
    - `data/processed/pmids_gold.txt`
    - `data/processed/pmids_distractors.txt`
- Robustness:
  - Allow resume/restart (write checkpoints while materializing)
  - Retry transient download/IO failures

#### FR-2.3 Reproducible manifest

Create `data/raw/manifest.json` that records exactly what went into the corpus:

- Corpus build timestamp
- Source method: `huggingface`
- Hugging Face dataset provenance (name, version, revision/commit, splits/shards)
- Sampling & filtering (`sampling_seed`, gold PMID inclusion rule, distractor sampling rule, filters)
- Counts (#PMIDs gold, #PMIDs distractors, #records materialized, #chunks produced)
- Integrity (SHA256 checksums of `corpus.jsonl`, `chunks.jsonl`, PMID lists)

#### FR-2.4 Corpus inclusion rules

- Include all PMIDs referenced by **BioASQ gold documents**.
- Include all PMIDs referenced by **PubMedQA** examples.
- Add a distractor set sampled with a fixed seed:
  - **Default size: 10k–20k abstracts** (sufficient for this project scope; fits comfortably in RAM with FAISS)
  - Configurable via `distractor_count` parameter

**Output format:**

```json
{"pmid":"12345678","title":"...","abstract":"...","year":2020,"source":"pubmed"}
```

### 3.3 [FR-3] Chunking (optimization target)

Implement chunking as a pluggable component with configurations:

- **Chunker types:**
  - Sentence-aware splitter (recommended)
  - Token/character splitter (baseline)
- **Parameters (sweepable):**
  - `chunk_size`
  - `chunk_overlap`
  - `separators` (if using recursive splitters)
- **Store chunk metadata:**
  - `pmid`, `chunk_id`, offsets, section tags if available

### 3.4 [FR-4] Embedding (pluggable)

- Interface: `embed_documents(chunks)` and `embed_query(question)`
- Caching: persist embeddings keyed by `(model_name, chunk_hash)`.

### 3.5 [FR-5] Vector Store & Indexing (FAISS-only)

- Build FAISS index over chunk embeddings.
- Persist:
  - FAISS index file
  - Chunk metadata store (e.g., SQLite / Parquet / JSONL)
- Support rebuilding the index deterministically from the corpus + config.

### 3.6 [FR-6] Retrieval (optimization target)

Implement retriever modes using the LangChain FAISS retriever interface.

#### Modes

- `similarity` (top-k)
- `mmr` (Maximal Marginal Relevance)
- `similarity_score_threshold` (or equivalent score-threshold filtering)

#### Sweepable parameters

- `k`
- `fetch_k` (for MMR)
- `lambda_mult` (MMR diversity)
- `score_threshold` (if using threshold mode)

#### Output schema

Ranked `Document` chunks with:
- `page_content`
- `metadata` (must include `pmid`, `chunk_id`, and any provenance fields)
- Retrieval score (store separately if the VectorStore wrapper doesn't attach it)

### 3.7 [FR-7] Reranking (required; optimization target)

Reranking is **required** for this project. It provides significant quality gains and is a core part of the RAG optimization story.

#### GPU-friendly reranker options

- **Cross-encoder reranker (HuggingFace)** over top-N retrieved chunks — **recommended default**
- LLM reranker via OpenAI (more expensive; use sparingly)

#### Sweepable parameters

- Reranker model name
- `top_n` to rerank
- Final `k` for generation evidence

### 3.8 [FR-8] Prompting & Generation (optimization target)

Prompt templates must be configurable and versioned.

#### Requirements

- Answer must be grounded in provided evidence
- Include citations (PMID + chunk_id)
- Refuse/abstain when evidence is insufficient
- Output a normalized schema (see below)

#### FR-8.1 Parseable outputs (required)

To guarantee a stable evaluation pipeline, generation output must be **machine-parseable**.

- Use **OpenAI structured outputs / JSON mode** (or tool/function calling) to enforce the response schema.
- Validate with a strict schema (e.g., Pydantic). On validation failure:
  - Retry generation up to N times (configurable)
  - If still invalid, mark the sample as `answer_type="unknown"` and log an artifact

#### FR-8.2 Strict citation policy (required)

- The model must cite **only from retrieved evidence chunks**.
- Policy options (make this a prompt/config toggle you can sweep):
  - **Strict:** every sentence must include ≥1 citation
  - **Claim-level:** each factual claim group includes ≥1 citation
- Citations must point to `pmid` + `chunk_id` present in the evidence set.

#### FR-8.3 Abstain/refusal trigger (required)

Define explicit abstention logic (configurable) so you can measure and tune it.

- **Evidence thresholding:** abstain if the top retrieval score is below `min_evidence_score` (or if fewer than `min_evidence_chunks` are above a threshold).
- **Model self-check:** optionally ask the model to output a `supported=true|false` flag based strictly on provided evidence; abstain if `supported=false`.
- Always log why abstention occurred (score-based, self-check, or both).

#### Suggested output schema

```json
{
  "question_id": "bioasq_001",
  "question": "...",
  "dataset": "bioasq|pubmedqa",
  "answer_type": "factoid|list|yesno|summary|unknown",
  "answer": "...",
  "citations": [{"pmid":"12345678","chunk_id":"pmid:12345678#c03"}],
  "confidence": 0.0,
  "debug": {
    "chunking": {"type":"recursive","chunk_size":350,"overlap":40},
    "retriever": {"mode":"mmr","k":10,"fetch_k":50,"lambda_mult":0.5},
    "reranker": {"enabled":true,"model":"...","top_n":50},
    "latency_ms": {"retrieve": 120, "rerank": 80, "generate": 900}
  }
}
```

### 3.9 [FR-9] Evaluation Harness (Golden Tests)

#### BioASQ metrics

**Retrieval:**
- Recall@k (required)
- MRR (required)
- Optional: Precision@k, MAP

**Answer:**
- Yes/No: accuracy
- Factoid: exact match (EM), token-F1
- List: set-F1
- Summary (ideal): ROUGE-L (required) and optional BERTScore

#### PubMedQA metrics

- Label prediction accuracy (required)
- Macro-F1 (recommended)

### 3.10 [FR-10] Experiment Runner — RapidFire AI (core differentiator)

Build a sweep system using **[RapidFire AI](https://github.com/RapidFireAI/rapidfireai)** to run many RAG configurations and produce a leaderboard.

#### FR-10.1 RapidFire AI integration (required)

RapidFire AI is the **primary sweep driver** for this project. It provides:

- **Hyperparallelized execution**: 16-24x throughput improvement over sequential runs
- **Shard-based scheduling**: compare many configurations concurrently, even on a single GPU
- **Interactive control**: stop, resume, clone-modify runs in real-time
- **Automatic optimization**: intelligent GPU utilization and rate limit management for OpenAI API

Run sweeps locally on the development machine using RapidFire AI.

#### FR-10.2 Sweep configuration

- Config-driven runs (YAML or TOML)
- Support grid sweeps over:
  - Chunking params
  - Retriever params
  - Reranker params
  - Prompt template variants

#### FR-10.3 Output

- Metrics summary table (CSV/JSON)
- Per-run artifact bundle (retrieval results, prompts, outputs)
- "Best config" report identifying top-performing configurations
- Leaderboard (`runs/leaderboard.csv`) comparing all sweep runs

### 3.11 [FR-11] API + CLI

#### CLI commands

| Command | Description |
|---------|-------------|
| `ingest_bioasq` | Load BioASQ dataset |
| `ingest_pubmedqa` | Load PubMedQA dataset |
| `build_corpus` | Build corpus from PubMed abstracts |
| `index_faiss` | Build FAISS index |
| `eval` | Run evaluation on golden suite |
| `sweep` | Run RapidFire AI sweep |
| `serve` | Start API server for demo |

#### API endpoints (for Gradio demo backend)

| Endpoint | Description |
|----------|-------------|
| `POST /answer` | Answer a question with citations |
| `POST /retrieve` | Retrieve chunks without generation |
| `GET /health` | Health check |

> **Note:** No authentication required. The demo is publicly accessible.

---

## 4. Non-Functional Requirements

> **Convention:** Non-functional requirements are labeled **[NFR-#]** — these define *how* the system must behave (quality attributes).

### 4.1 [NFR-1] Reproducibility

- Every run must produce a `run.json` containing:
  - Git commit SHA
  - Config used
  - Model names (LLM + embeddings + reranker)
  - Dataset versions
  - Random seeds
- Deterministic chunking when given the same input + config.

### 4.2 [NFR-2] Performance & Iteration Speed

- Persist FAISS index and embeddings cache to avoid recompute.
- Batch evaluation jobs (vectorized embedding queries, batched reranking where possible).

### 4.3 [NFR-3] Cost Control (OpenAI)

- **Secrets handling:** load API keys from environment variables (e.g., `OPENAI_API_KEY`). Never commit keys; add `.env` to `.gitignore`.
- **Prompt-hash caching:** cache LLM outputs keyed by a stable hash of:
  - Model name
  - Prompt template version
  - Full rendered prompt (including evidence)
  - Decoding params (temperature, max tokens)
  
  This prevents re-paying for identical evaluations during sweeps.
- **Per-run budget guardrails (required):** enforce configurable caps:
  - `max_questions` (hard limit on examples evaluated)
  - `max_total_tokens` (input + output)
  - `max_usd` (estimated or measured)
- **Budget exceeded behavior (configurable):**
  - **fail-fast:** stop the run and mark it failed
  - **skip:** skip remaining questions and mark the run as partial (useful for exploratory sweeps)
- **Reporting:** every run must report total tokens, estimated cost, and cache hit rate.
- **Timeout handling:** configure a timeout for OpenAI API calls (default: 30 seconds). On timeout, retry up to N times before marking the sample as failed.

### 4.4 [NFR-4] Maintainability

- Clear module boundaries: `data/`, `chunking/`, `index/`, `retrieve/`, `rerank/`, `generate/`, `eval/`, `experiments/`
- Unit tests for scoring logic and config parsing.

### 4.5 [NFR-5] Safety

- Prominent disclaimer: for educational use only, not medical advice.
- Provide evidence-first output with citations; abstain when uncertain.

---

## 5. Technical Design

### 5.1 Tech Stack

#### Python + core libs

| Component | Technology |
|-----------|------------|
| Language | Python 3.12.x |
| RAG Framework | LangChain (`langchain-community`, `langchain-openai`) |
| Vector Store | FAISS (`faiss-cpu` or `faiss-gpu`) |
| ML Framework | PyTorch (for GPU rerankers) |
| API | FastAPI |
| Testing | pytest |
| Linting | ruff |
| Typing | mypy (optional) |

#### Front-end (demo web app)

| Component | Technology |
|-----------|------------|
| Framework | Gradio |
| Hosting | HuggingFace Spaces |
| Authentication | None (public demo) |

**Side-by-side comparison view** (key demo feature):
- Left panel: Baseline RAG configuration
- Right panel: Optimized RAG configuration
- Synced input: same question sent to both pipelines simultaneously

**Display for each response:**
- Answer with inline citations
- Retrieved chunks with scores (before/after rerank)
- Latency breakdown (retrieve / rerank / generate)
- Config summary (chunking, retriever, reranker settings)

#### OpenAI usage

- `ChatOpenAI` for generation
- Embeddings:
  - Option A: OpenAI embeddings (fast to start, consistent)
  - Option B: Local embedding model (optional stretch; can reduce cost)

> Keep the embeddings choice pluggable.

### 5.2 Architecture

#### High-level flow

```
1. Load corpus docs (PubMed abstracts)
2. Chunk docs → Document objects
3. Embed chunks → vectors
4. Index in FAISS (+ metadata store)
5. For each question:
   ├── Retrieve chunks (similarity/MMR/threshold)
   ├── Rerank top-N with cross-encoder
   ├── Build prompt with top evidence
   ├── Call OpenAI LLM
   └── Parse and normalize output
6. Score against gold labels (BioASQ / PubMedQA)
7. Save metrics + artifacts
```

### 5.3 Repo Layout

```
biorag-bench/
├── .gitignore
├── .env.example                      # Document required env vars (OPENAI_API_KEY)
├── README.md
├── pyproject.toml
├── Makefile                          # Common dev commands
│
├── configs/
│   ├── base.yaml                     # Default configuration
│   ├── prompts/
│   │   ├── cite_and_abstain_v1.txt
│   │   └── cite_and_abstain_v2.txt
│   └── sweeps/
│       ├── chunking_sweep.yaml
│       ├── retriever_sweep.yaml
│       └── full_sweep.yaml
│
├── data/
│   ├── raw/
│   │   ├── manifest.json             # Dataset provenance (FR-2.3)
│   │   ├── bioasq/
│   │   └── pubmedqa/
│   ├── processed/
│   │   ├── corpus/
│   │   │   ├── corpus.jsonl
│   │   │   └── chunks.jsonl
│   │   ├── pmids_gold.txt
│   │   ├── pmids_distractors.txt
│   │   └── embeddings/               # Cached embeddings (FR-4)
│   └── golden/
│       ├── bioasq_golden_200.jsonl   # Stable eval subset (2.4)
│       └── pubmedqa_golden_500.jsonl
│
├── runs/                             # Experiment outputs (gitignored except leaderboard)
│   ├── .gitkeep
│   └── leaderboard.csv               # Top configs comparison
│
├── src/
│   └── biorag/
│       ├── __init__.py
│       ├── py.typed                  # PEP 561 marker for mypy
│       │
│       ├── cli/                      # FR-11: CLI commands
│       │   ├── __init__.py
│       │   └── main.py               # Typer app: ingest, build_corpus, index, eval, sweep, serve
│       │
│       ├── schemas/                  # Pydantic models (FR-8.1)
│       │   ├── __init__.py
│       │   ├── config.py             # Config schema (chunking, retrieval, rerank, etc.)
│       │   ├── corpus.py             # CorpusDocument, Chunk
│       │   ├── evaluation.py         # EvalResult, RunMetrics
│       │   └── generation.py         # AnswerOutput (question_id, answer, citations, etc.)
│       │
│       ├── data/                     # FR-1, FR-2: Data loading
│       │   ├── __init__.py
│       │   ├── bioasq_loader.py
│       │   ├── pubmedqa_loader.py
│       │   └── corpus_builder.py     # FR-2: PubMed abstracts corpus
│       │
│       ├── chunking/                 # FR-3: Chunking strategies
│       │   ├── __init__.py
│       │   ├── base.py               # Abstract chunker interface
│       │   ├── recursive.py          # Sentence-aware splitter
│       │   └── token.py              # Token/character splitter (baseline)
│       │
│       ├── embeddings/               # FR-4: Embedding interface
│       │   ├── __init__.py
│       │   ├── base.py               # Abstract embedder interface
│       │   ├── openai.py             # OpenAI embeddings
│       │   ├── local.py              # Optional: local HF embeddings
│       │   └── cache.py              # Embedding cache (keyed by model + chunk hash)
│       │
│       ├── indexing/                 # FR-5: FAISS vector store
│       │   ├── __init__.py
│       │   ├── faiss_store.py        # FAISS index wrapper
│       │   └── metadata_store.py     # Chunk metadata (SQLite/Parquet)
│       │
│       ├── retrieve/                 # FR-6: Retrieval
│       │   ├── __init__.py
│       │   └── retriever.py          # similarity, mmr, threshold modes
│       │
│       ├── rerank/                   # FR-7: Reranking
│       │   ├── __init__.py
│       │   ├── base.py               # Abstract reranker interface
│       │   ├── cross_encoder.py      # HF cross-encoder (GPU)
│       │   └── llm_reranker.py       # Optional: OpenAI-based reranker
│       │
│       ├── generate/                 # FR-8: Prompting & generation
│       │   ├── __init__.py
│       │   ├── prompts.py            # Prompt template loader
│       │   ├── generator.py          # LLM generation with structured output
│       │   └── abstention.py         # Abstain/refusal logic (FR-8.3)
│       │
│       ├── eval/                     # FR-9: Evaluation harness
│       │   ├── __init__.py
│       │   ├── metrics.py            # Recall@k, MRR, EM, F1, ROUGE-L, etc.
│       │   ├── bioasq_eval.py        # BioASQ-specific scoring
│       │   ├── pubmedqa_eval.py      # PubMedQA label accuracy
│       │   └── harness.py            # Unified evaluation runner
│       │
│       ├── experiments/              # FR-10: Experiment runner
│       │   ├── __init__.py
│       │   ├── runner.py             # Single run executor
│       │   ├── sweep.py              # RapidFire AI sweep integration
│       │   └── artifacts.py          # Run artifact management (run.json, etc.)
│       │
│       ├── api/                      # FR-11: FastAPI endpoints
│       │   ├── __init__.py
│       │   ├── app.py                # FastAPI app factory
│       │   ├── routes.py             # /answer, /retrieve, /health
│       │   └── dependencies.py       # Pipeline injection
│       │
│       ├── pipeline/                 # End-to-end RAG pipeline orchestration
│       │   ├── __init__.py
│       │   └── rag.py                # RAGPipeline class (retrieve → rerank → generate)
│       │
│       └── utils/
│           ├── __init__.py
│           ├── caching.py            # LLM output cache (NFR-3)
│           ├── cost.py               # Token counting, budget guardrails
│           └── logging.py            # Structured logging
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Shared fixtures
│   ├── unit/
│   │   ├── test_chunking.py
│   │   ├── test_metrics.py
│   │   ├── test_schemas.py
│   │   └── test_config.py
│   └── integration/
│       ├── test_retrieval.py
│       └── test_pipeline.py
│
├── demo/                             # Gradio app for HuggingFace Spaces
│   ├── app.py                        # Main Gradio interface
│   ├── requirements.txt              # Minimal deps for Spaces
│   └── README.md                     # Spaces-specific instructions
│
├── scripts/                          # One-off utility scripts
│   ├── download_datasets.py          # Initial data download
│   └── validate_golden_suite.py      # Sanity check golden files
│
└── notebooks/                        # Optional: exploration & analysis
    ├── 01_data_exploration.ipynb
    └── 02_failure_analysis.ipynb
```

#### Key design decisions

| Addition | Rationale |
|----------|-----------|
| **`cli/`** | FR-11 defines 7 CLI commands — they need a dedicated module |
| **`schemas/`** | FR-8.1 requires strict Pydantic models for LLM outputs; centralizes all data contracts |
| **`pipeline/`** | Clean orchestration layer that composes retrieve → rerank → generate |
| **`indexing/`** | Renamed from `index/` for clarity; avoids confusion with Python's index concept |
| **`data/golden/`** | Explicit location for the stable evaluation subset (Section 2.4) |
| **`py.typed`** | PEP 561 compliance for mypy (listed in tech stack) |
| **`.env.example`** | Documents `OPENAI_API_KEY` requirement (NFR-3) |
| **`Makefile`** | Convenient shortcuts: `make test`, `make lint`, `make eval`, etc. |
| **`scripts/`** | One-off tasks that don't fit in the main package |
| **`notebooks/`** | Useful for failure analysis deliverable (Section 6.2) |
| **`tests/unit/` + `tests/integration/`** | Better test organization as project grows |

### 5.4 Config Design

Example: `configs/base.yaml`

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 350

embeddings:
  provider: openai
  model: text-embedding-3-large

chunking:
  type: recursive
  chunk_size: 350
  chunk_overlap: 40

retrieval:
  mode: mmr
  k: 10
  fetch_k: 50
  lambda_mult: 0.5

rerank:
  enabled: true
  model: cross-encoder
  top_n: 50
  final_k: 8

prompt:
  template: prompts/cite_and_abstain_v2.txt
```

### 5.5 GPU Usage

Your GPU is used for:

| Use Case | Required? |
|----------|-----------|
| **Cross-encoder reranking** | Required — significant quality gains |
| Local embeddings | Optional — cost reduction / faster iteration |

Keep the baseline working with OpenAI embeddings first; add local embeddings only if time allows.

---

## 6. Deliverables

### 6.1 Required

#### Public repo

- `README` quickstart
- FAISS indexing scripts
- Evaluation harness for BioASQ + PubMedQA
- RapidFire AI sweep runner producing a metrics leaderboard

#### Gradio demo on HuggingFace Spaces

- Side-by-side comparison (baseline vs optimized RAG)
- Public access (no authentication)

### 6.2 Valuable Add-ons

Within the 2–4 week timeline:

- A "Top configs" leaderboard table committed to repo (`runs/leaderboard.csv`)
- A short failure analysis doc:
  - Show 3–5 examples where retrieval failed
  - Show prompt/rerank fix that improved it
- A cost/latency report:
  - Average tokens per answer
  - Retrieval/rerank/generate latency

---

*Document version:* v4 (FR/NFR labeling convention)
