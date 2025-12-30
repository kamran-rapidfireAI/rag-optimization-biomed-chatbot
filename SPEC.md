# Project: BioRAG Bench (LangChain Edition) — Evidence-Grounded Biomedical QA with Golden Tests (BioASQ + PubMedQA)

> **Elevator pitch:** Build a modular biomedical Retrieval-Augmented Generation (RAG) pipeline using **LangChain components** (FAISS vector store, configurable chunking, retriever settings, optional reranking, prompt variants) and evaluate it with **true golden tests** from **BioASQ Task B** and **PubMedQA** — **no manual labeling required**.

---

## 0) Your chosen constraints (baked into this spec)

- **LLM:** Hosted API (**OpenAI**) for generation (and optionally embeddings).
- **Indexing:** **FAISS only** (keep infra simple; no OpenSearch/Elastic).
- **Timeline:** **2–4 weeks** (portfolio-quality, not “research project”).
- **Hardware:** You have a **GPU** (useful for local rerankers / cross-encoders).
- **Focus:** **Optimization** via systematic sweeps:
  - chunking strategy + parameters
  - retriever parameters (k, search type, MMR, score thresholds)
  - reranker parameters/models
  - prompt templates / citation formats / refusal policy
- **Benchmarks:** **BioASQ + PubMedQA**.

---

## 1) Goals

### Primary goals

1. **End-to-end RAG pipeline**: ingest → chunk → embed → index (FAISS) → retrieve → (optional rerank) → generate → cite sources.
2. **Golden tests / automatic evaluation** using public labeled benchmarks:
   - BioASQ: gold relevant docs/snippets + gold answers (exact & ideal)
   - PubMedQA: gold label (yes/no/maybe) and supporting context
3. **Optimization-first workflow**:
   - Run parameter sweeps and track metrics + artifacts
   - CI regression gates on a stable golden subset (e.g., 200 questions)

### Secondary goals

- Minimal API service for demo (FastAPI)
- Reproducible experiment tracking (run configs + metrics + artifacts)
- Fast iteration: caching embeddings, persistent FAISS index, batch evaluation.

### Non-goals

- Medical advice
- State-of-the-art leaderboard chasing (you’re demonstrating engineering + measurement)
- Operating a large search cluster (kept intentionally simple).

---

## 2) Tech stack (recommended)

### Python + core libs

- Python 3.12.x
- LangChain (plus `langchain-community`, `langchain-openai`)
- FAISS (`faiss-cpu` or `faiss-gpu`)
- PyTorch (for GPU rerankers)
- FastAPI (optional serving)
- pytest (tests), ruff (lint), mypy (optional typing)

### Front-end (optional demo web app)

- **Next.js (React) + TypeScript**
- **Tailwind CSS** for styling (optionally `shadcn/ui` components)
- Calls the FastAPI backend via JSON:
  - `POST /answer`
  - `POST /retrieve`
- UI focuses on debug/iteration features:
  - show retrieved chunks + scores (before/after rerank)
  - show final prompt (or a redacted version) and model output
  - show citations and latency breakdown
  - dropdown to pick a saved run/config for side-by-side comparison

### OpenAI usage

- `ChatOpenAI` for generation
- embeddings:
  - Option A: OpenAI embeddings (fast to start, consistent)
  - Option B: local embedding model (optional stretch; can reduce cost)

> Keep embeddings choice pluggable;

---

## 2.5 Intended chatbot questions (scope)

This is an **evidence-grounded biomedical literature QA chatbot** (a “PubMed literature assistant”), not a general conversational assistant.

### Best-fit question types (aligned with BioASQ + PubMedQA)

1) **Yes/No/Maybe questions** (strong fit; easy to score)
- *Example:* “Does metformin reduce cancer risk in patients with diabetes?”
- *Example:* “Is vitamin D supplementation associated with reduced fracture risk in older adults?”

2) **Factoid questions** (short, specific answers)
- *Example:* “Which gene is mutated in Huntington’s disease?”
- *Example:* “What receptor does naloxone primarily antagonize?”

3) **List questions** (return a set of items with citations)
- *Example:* “What are common adverse effects of amiodarone?”
- *Example:* “Which drugs are ACE inhibitors?”

4) **Summary / synthesis questions** (best for showing RAG value + citations)
- *Example:* “Summarize evidence on SGLT2 inhibitors and heart failure outcomes.”
- *Example:* “What does the literature say about ketogenic diets and epilepsy control?”

### Out-of-scope queries (must refuse / reframe)

- Personalized medical advice:
  - “Should I take X?” “What dose should I take?” “Is this safe for me?”
- Diagnosis / treatment planning for an individual

**Reframe pattern:** “I can’t give personal medical advice, but I can summarize what published studies report about X for condition Y and cite the relevant PubMed articles.”

---

## 3) Datasets and “golden tests”

### 3.1 BioASQ Task B (required)

Per question you get:

- gold relevant **documents** (PMIDs) and/or **snippets**
- gold answers:
  - **exact** answers (factoid/list/yes-no)
  - **ideal** answers (summary paragraph)

These labels power both retrieval and generation evaluation.

### 3.2 PubMedQA (required)

Per sample you get:

- question (usually yes/no/maybe style)
- label: **yes/no/maybe**
- supporting abstract/context

This provides a second benchmark with clean auto-scoring.

### 3.3 Dataset sources (exact download locations)

> Add these links to your README and also record the **exact version/date** of each download in `data/raw/manifest.json`.

**BioASQ Task B datasets (official)**

- BioASQ Participants Area → **Datasets** (Task B / “Task *b*” downloads)
- BioASQ Participants Area → Task *b* page (includes dataset JSON format notes)

**PubMedQA datasets (official)**

- PubMedQA homepage (links to the dataset + code repository)
- PubMedQA GitHub repository (download instructions and splitting scripts)

**Convenience mirrors (optional)**

- Hugging Face datasets: `bigbio/bioasq_task_b` (community packaging)
- Hugging Face datasets: `qiaojin/PubMedQA` (community packaging)

**PubMed abstracts for your retrieval corpus**

- **Chosen approach:** build the retrieval corpus from a **Hugging Face PubMed abstracts dataset**.
- Record the **dataset name**, **dataset version**, and **revision/commit hash** in `data/raw/manifest.json`.

### 3.4 Clarification: what “comes from PubMed”

This project uses **PubMed** in two different ways:

- **Underlying literature source (corpus):** the retrieval corpus is built from **PubMed records (titles/abstracts)**. PubMed is the original source of that text.
- **Benchmark datasets (labels/tests):** BioASQ and PubMedQA are **separate benchmark datasets** created by their respective organizers/authors. They *reference* PubMed articles (PMIDs) and are grounded in PubMed literature, but they are not “PubMed datasets” themselves.
- **Hugging Face role:** Hugging Face is the **distribution/packaging layer** used to download (a) the benchmark datasets and (b) a PubMed-abstracts corpus.

### 3.5 Who created BioASQ and PubMedQA

- **BioASQ Task B:** created/curated by the **BioASQ challenge organizers** (a consortium-supported effort) with **biomedical experts** writing questions and providing gold documents/snippets and gold answers.
- **PubMedQA:** created by the authors of the PubMedQA dataset/paper (**Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W. Cohen, Xinghua Lu**).

---

## 4) Functional requirements

---

## 4) Functional requirements

### R1. Data loaders

- **BioASQ loader**
  - Parse questions, types, gold docs/snippets, exact/ideal answers
  - Assign stable `question_id` keys for caching and evaluation artifacts
- **PubMedQA loader**
  - Parse train/val/test splits
  - Extract `question_id`, question, label, and supporting context/PMIDs (if available)

### R2. Corpus builder (PubMed abstracts)

Build a corpus of texts to retrieve from.

#### R2.1 Source of abstracts (selected: **Option A — Hugging Face**)

**Option A — Hugging Face (selected for this project)**

- Use a Hugging Face dataset that contains PubMed abstracts at scale.
- Pin the **exact dataset revision** (commit hash) so the corpus is reproducible.
- Sample a **distractor set** from the same dataset using a fixed random seed.
- **Important practical note:** the popular `ncbi/pubmed` dataset represents the full PubMed baseline and is **very large** (tens of millions of records). Treat it as an optional source for *distractors* or large-scale experiments; for a 2–4 week project, prefer a smaller HF PubMed corpus.

> **Decision:** This project will use **Option A (Hugging Face)** to build the retrieval corpus.

**Option B — NCBI E-utilities (PMID-driven, alternative)**

- Build a PMID list (all BioASQ gold PMIDs + all PubMedQA PMIDs + distractor PMIDs).
- Fetch abstracts via NCBI E-utilities (e.g., `esummary`/`efetch`) with:
  - on-disk caching
  - retries with exponential backoff
  - a polite rate limit (configurable; default 3–10 req/s depending on your API key)

> **Note:** Option B is kept as a documented alternative. The project baseline uses **Option A (Hugging Face)**.

#### R2.2 Caching and reproducible downloads (required)

Because the baseline corpus source is **Hugging Face**, this section is about *repeatable dataset materialization* rather than NCBI request throttling.

- Cache at two layers:
  1. **HF dataset cache**: rely on the standard Hugging Face datasets cache (do not commit it). Record dataset **name + revision** (see manifest) so it can be rehydrated.
  2. **Materialized corpus cache**: write a deterministic, normalized `corpus.jsonl` under `data/processed/corpus/` (and a `chunks.jsonl` after chunking).
- Deterministic materialization:
  - use a fixed `sampling_seed`
  - record the exact split/shard selection logic
  - persist the exact **PMID lists** used:
    - `data/processed/pmids_gold.txt`
    - `data/processed/pmids_distractors.txt`
- Robustness:
  - allow resume/restart (write checkpoints while materializing)
  - retry transient download/IO failures

#### R2.3 Reproducible manifest (required)

Create `data/raw/manifest.json` that records exactly what went into the corpus, including:

- corpus build timestamp
- source method: `huggingface`
- Hugging Face dataset provenance:
  - dataset name (e.g., `ncbi/pubmed` or a smaller PubMed abstracts dataset)
  - dataset version
  - **dataset revision/commit** (pin this)
  - selected splits/shards (if applicable)
- sampling & filtering:
  - `sampling_seed`
  - gold PMID inclusion rule (BioASQ ∪ PubMedQA)
  - distractor sampling rule (how many, from where)
  - any language/abstract-present filters
- counts:
  - \#PMIDs (gold)
  - \#PMIDs (distractors)
  - \#records materialized
  - \#chunks produced
- integrity:
  - checksum (e.g., SHA256) of final `corpus.jsonl`
  - checksum of final `chunks.jsonl`
  - checksum of `pmids_gold.txt` and `pmids_distractors.txt`

#### R2.4 Corpus inclusion rules (minimum viable)

- Include all PMIDs referenced by **BioASQ gold documents**.
- Include all PMIDs referenced by **PubMedQA** examples.
- Add a configurable distractor set (e.g., 10k–100k abstracts) sampled with a fixed seed.

Output format (example):

```json
{"pmid":"12345678","title":"...","abstract":"...","year":2020,"source":"pubmed"}
```

### R3. Chunking (optimization target)

Implement chunking as a pluggable component with configurations:

- chunker types:
  - sentence-aware splitter (recommended)
  - token/character splitter (baseline)
- parameters (sweepable):
  - `chunk_size`
  - `chunk_overlap`
  - `separators` (if using recursive splitters)
- store chunk metadata:
  - `pmid`, `chunk_id`, offsets, section tags if available

### R4. Embedding (pluggable)

- Interface: `embed_documents(chunks)` and `embed_query(question)`
- Caching: persist embeddings keyed by `(model_name, chunk_hash)`.

### R5. Vector store & indexing (FAISS-only)

- Build FAISS index over chunk embeddings.
- Persist:
  - FAISS index file
  - chunk metadata store (e.g., SQLite / Parquet / JSONL)
- Support rebuilding index deterministically from corpus + config.

### R6. Retrieval (optimization target)

Implement retriever modes using the LangChain FAISS retriever interface.

#### R6.1 Modes

- `similarity` (top-k)
- `mmr` (Maximal Marginal Relevance)
- `similarity_score_threshold` (or equivalent score-threshold filtering)

#### R6.2 Sweepable parameters

- `k`
- `fetch_k` (for MMR)
- `lambda_mult` (MMR diversity)
- `score_threshold` (if using threshold mode)

#### R6.3 Output schema

- ranked `Document` chunks with:
  - `page_content`
  - `metadata` (must include `pmid`, `chunk_id`, and any provenance fields)
  - retrieval score (store separately if the VectorStore wrapper doesn’t attach it)

### R7. Reranking (optional but recommended; optimization target)

GPU-friendly reranker options:

- Cross-encoder reranker (HuggingFace) over top-N retrieved chunks
- (Optional) LLM reranker via OpenAI (more expensive; use sparingly)

Sweepable parameters:

- reranker model name
- `top_n` to rerank
- final `k` for generation evidence

### R8. Prompting & generation (optimization target)

- Prompt templates must be configurable and versioned.
- Requirements:
  - Answer must be grounded in provided evidence
  - Include citations (PMID + chunk\_id)
  - Refuse/abstain when evidence is insufficient
  - Output a normalized schema (see below)

#### R8.1 Parseable outputs (required)

To guarantee a stable evaluation pipeline, generation output must be **machine-parseable**.

- Use **OpenAI structured outputs / JSON mode** (or tool/function calling) to enforce the response schema.
- Validate with a strict schema (e.g., Pydantic). On validation failure:
  - retry generation up to N times (configurable)
  - if still invalid, mark the sample as `answer_type="unknown"` and log an artifact

#### R8.2 Strict citation policy (required)

- The model must cite **only from retrieved evidence chunks**.
- Policy options (make this a prompt/config toggle you can sweep):
  - **Strict:** every sentence must include ≥1 citation
  - **Claim-level:** each factual claim group includes ≥1 citation
- Citations must point to `pmid` + `chunk_id` present in the evidence set.

#### R8.3 Abstain/refusal trigger (required)

Define explicit abstention logic (configurable) so you can measure and tune it.

- **Evidence thresholding:** abstain if the top retrieval score is below `min_evidence_score` (or if fewer than `min_evidence_chunks` are above a threshold).
- **Model self-check:** optionally ask the model to output a `supported=true|false` flag (or a short rationale) based strictly on provided evidence; abstain if `supported=false`.
- Always log why abstention occurred (score-based, self-check, or both).

Suggested output schema:

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

### R9. Evaluation harness (golden tests)

#### BioASQ

**Retrieval metrics**

- Recall\@k (must)
- MRR (must)
- Optional: Precision\@k, MAP

**Answer metrics**

- Yes/No: accuracy
- Factoid: exact match (EM), token-F1
- List: set-F1
- Summary (ideal): ROUGE-L (must) and optional BERTScore

#### PubMedQA

- Label prediction accuracy (must)
- Macro-F1 (recommended)

> For PubMedQA, decide whether you treat it as:
>
> - a direct “label prediction” task, or
> - a RAG + evidence task that still outputs yes/no/maybe with citations. This project assumes the second (RAG + citations) for consistency.

### R10. Experiment runner (core differentiator)

Build a sweep system that can run many RAG configurations and produce a leaderboard.

Requirements:

- Config-driven runs (YAML or TOML)
- Support grid sweeps over:
  - chunking params
  - retriever params
  - reranker params
  - prompt template variants
- Output:
  - metrics summary table (CSV/JSON)
  - per-run artifact bundle (retrieval results, prompts, outputs)
  - “best config” report

Optional (nice-to-have):

- Integrate **RapidFireAI** to drive sweeps / benchmarking if you want faster config exploration.

### R11. API + CLI

- CLI commands:

  - `ingest_bioasq`
  - `ingest_pubmedqa`
  - `build_corpus`
  - `index_faiss`
  - `eval`
  - `sweep`
  - `serve` (optional)

- FastAPI endpoints (optional):

  - `POST /answer`
  - `POST /retrieve`
  - `GET /health`

---

## 5) Non-functional requirements

### N1. Reproducibility

- Every run must produce a `run.json` containing:
  - git commit SHA
  - config used
  - model names (LLM + embeddings + reranker)
  - dataset versions
  - random seeds
- Deterministic chunking when given same input + config.

### N2. Performance & iteration speed

- Persist FAISS index and embeddings cache to avoid recompute.
- Batch evaluation jobs (vectorized embedding queries, batched reranking where possible).

### N3. Cost control (OpenAI)

- **Secrets handling:** load API keys from environment variables (e.g., `OPENAI_API_KEY`). Never commit keys; add `.env` to `.gitignore`.
- **Prompt-hash caching:** cache LLM outputs keyed by a stable hash of:
  - model name
  - prompt template version
  - full rendered prompt (including evidence)
  - decoding params (temperature, max tokens) This prevents re-paying for identical evaluations during sweeps.
- **Per-run budget guardrails (required):** enforce configurable caps:
  - `max_questions` (hard limit on examples evaluated)
  - `max_total_tokens` (input + output)
  - `max_usd` (estimated or measured)
- **Budget exceeded behavior (configurable):**
  - **fail-fast:** stop the run and mark it failed (recommended for CI)
  - **skip:** skip remaining questions and mark the run as partial (useful for exploratory sweeps)
- **Reporting:** every run must report total tokens, estimated cost, and cache hit rate.

### N4. Maintainability

- Clear module boundaries: `data/`, `chunking/`, `index/`, `retrieve/`, `rerank/`, `generate/`, `eval/`, `experiments/`
- Unit tests for scoring logic and config parsing.

### N5. Safety

- Prominent disclaimer: educational use only, not medical advice
- Evidence-first output with citations; abstain when uncertain.

---

## 6) Architecture (LangChain-first)

### High-level flow

1. Load corpus docs (PubMed abstracts)
2. Chunk docs → `Document` objects
3. Embed chunks → vectors
4. Index in FAISS (+ metadata store)
5. For each question:
   - retrieve chunks (similarity/MMR/threshold)
   - optional rerank top-N
   - build prompt with top evidence
   - call OpenAI LLM
   - parse and normalize output
6. Score against gold labels (BioASQ / PubMedQA)
7. Save metrics + artifacts

### Suggested repo layout

```
biorag-bench/
  README.md
  pyproject.toml
  configs/
    base.yaml
    prompts/
    sweeps/
  data/
    raw/
    processed/
  runs/                 # run outputs (metrics + artifacts)
  src/
    biorag/
      data/
      chunking/
      embeddings/
      index/
      retrieve/
      rerank/
      generate/
      eval/
      experiments/
      api/
      utils/
  tests/
  .github/workflows/
```

---

## 7) “Golden suite” and CI regression gates

### Golden suite

- Define a stable subset:
  - `bioasq_golden_200.jsonl`
  - `pubmedqa_golden_500.jsonl` (or 200–500 depending on runtime)
- Keep the subset deterministic (seeded sampling, committed to repo).

### CI gating (example)

CI fails if (relative to baseline JSON committed in repo):

- BioASQ Recall\@10 drops by > 1.0 point
- BioASQ MRR drops by > 0.5 point
- BioASQ exact-answer F1 drops by > 1.0 point
- PubMedQA accuracy drops by > 1.0 point

> Store baseline metrics in `configs/baselines/` and compare in CI.

---

## 8) Deliverables (what you will ship)

### Required

- Public repo with:
  - `README` quickstart
  - FAISS indexing scripts
  - evaluation harness for BioASQ + PubMedQA
  - sweep runner producing a metrics leaderboard
  - CI regression gates on golden suites

### Valuable add-ons (within 2–4 weeks)

- A “Top configs” leaderboard table committed to repo (`runs/leaderboard.csv`)
- A short failure analysis doc:
  - show 3–5 examples where retrieval failed
  - show prompt/rerank fix that improved it
- A cost/latency report:
  - average tokens per answer
  - retrieval/rerank/generate latency

---

## 9) Milestones (2–4 week plan)

### Week 1 — Baseline system + eval

- Ingest BioASQ + PubMedQA
- Build corpus from referenced PMIDs (+ distractors)
- Chunk + embed + FAISS index
- Basic retrieval + OpenAI generation + citations
- Implement evaluation for:
  - BioASQ retrieval (Recall\@k, MRR)
  - BioASQ answers (EM/F1/ROUGE-L)
  - PubMedQA label accuracy

### Week 2 — Optimization loop

- Add:
  - multiple chunkers + parameters
  - MMR + threshold retrieval
- Add experiment runner (`sweep`) and leaderboard output
- Establish golden suites and CI gating

### Week 3 — Reranking + better prompting (optional but high ROI)

- Add GPU cross-encoder reranker (top-N rerank)
- Improve prompts (citation discipline, abstain rules)
- Add artifact viewer scripts (per-question debug dumps)

### Week 4 — Polish

- Tighten docs, final leaderboard, failure analysis
- Optional demo API (FastAPI) + minimal UI

---

## 10) Config design (example)

`configs/base.yaml` (illustrative)

```yaml
llm:
  provider: openai
  model: gpt-4.1-mini
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

---

## 11) Notes on GPU usage

Your GPU is most valuable for:

- cross-encoder reranking (significant quality gains)
- optional local embeddings (cost reduction / faster iteration later)

Keep the baseline working with OpenAI embeddings first; add local embeddings only if time allows.

---

*Document version:* v1 (LangChain + FAISS + OpenAI + optimization focus + BioASQ+PubMedQA)

