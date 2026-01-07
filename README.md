# 🧬 BioRAG Bench

Benchmark-driven biomedical RAG optimization pipeline using LangChain, FAISS, and OpenAI. Evaluates retrieval-augmented generation on BioASQ and PubMedQA golden test suites.

[![HuggingFace Spaces](https://img.shields.io/badge/🤗-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/yourusername/biorag-bench)
[![Tests](https://img.shields.io/badge/tests-407%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-74.50%25-green)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()

> ⚠️ **Medical Disclaimer:** This is an educational/research project. It should NOT be used for medical diagnosis, treatment decisions, or as a substitute for professional medical advice.

---

## ✨ Features

- **End-to-end RAG Pipeline**: Retrieve → Rerank → Generate with structured JSON outputs
- **Side-by-Side Comparison**: Compare Baseline vs Optimized configurations in the demo
- **Biomedical Focus**: Optimized for PubMed literature and medical Q&A
- **Multiple Retrieval Modes**: Similarity, MMR, and threshold-based retrieval
- **GPU-Accelerated Reranking**: Cross-encoder reranking with CUDA support
- **Citation Enforcement**: Every answer includes verifiable PMID citations
- **Abstention Logic**: Model refuses to answer when evidence is insufficient
- **Cost Controls**: Token limits, budget caps, and LLM output caching
- **Parameter Sweeps**: RapidFire AI integration for hyperparameter optimization
- **FastAPI Backend**: Production-ready REST API with OpenAPI docs
- **Gradio Demo**: Interactive web interface with side-by-side comparison

---

## 📋 Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Ubuntu 20.04+, Windows 10/11, macOS 12+ | Ubuntu 22.04 |
| **CPU** | 8 cores | 12-16 cores |
| **RAM** | 16 GB | 32 GB |
| **GPU** | NVIDIA GPU with 8GB VRAM | RTX 3080/4070 (12GB+ VRAM) |
| **Storage** | 50 GB SSD | 256 GB NVMe SSD |

> ⚠️ **GPU Recommended:** Cross-encoder reranking is 10-20x faster on GPU (~15ms vs ~300ms on CPU).

### Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| Python | 3.11+ | 3.12 recommended |
| CUDA | 11.8+ | For GPU acceleration |
| Git | 2.30+ | Version control |

### API Keys

You will need an **OpenAI API key** for embeddings and LLM generation:

```bash
# Create .env file from template
cp env.example .env

# Edit .env and add your API key
OPENAI_API_KEY=sk-your-key-here
```

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-org/rag-optimization-biomed-chatbot.git
cd rag-optimization-biomed-chatbot

# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -e .
```

### 2. Configure Environment

```bash
# Copy environment template
cp env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### 3. Verify Installation

```bash
# Check CLI is working
biorag --help

# Show current configuration
biorag info

# Run test suite
python -m pytest tests/ -q
```

---

## 🧪 Testing

### Run All Tests

```bash
# Quick test run
python -m pytest tests/ -q

# With coverage report
python -m pytest tests/ --cov=src/biorag --cov-report=term-missing

# Run only unit tests
python -m pytest tests/unit/ -v

# Run only integration tests
python -m pytest tests/integration/ -v
```

### Current Test Status

- **407 tests passing**
- **74.50% code coverage**

---

## 🖥️ Running the Application

### Option 1: FastAPI Server

```bash
# Start the API server
biorag serve

# With custom options
biorag serve --host 0.0.0.0 --port 8000 --reload

# API will be available at:
# - API Root: http://localhost:8000/
# - Swagger Docs: http://localhost:8000/docs
# - Health Check: http://localhost:8000/api/v1/health
```

### Option 2: Gradio Demo

```bash
# Run the Gradio demo
python demo/app.py

# With options
python demo/app.py --host 0.0.0.0 --port 7860 --share

# Demo will be available at http://localhost:7860
```

The demo features:
- **Side-by-Side Comparison**: Compare Baseline vs Optimized RAG configurations
- **Single Query Mode**: Standard single-pipeline queries
- **Detailed Output**: Answer, citations, retrieved evidence, latency breakdown

### Option 3: Python API

```python
from biorag.pipeline import RAGPipeline
from biorag.schemas.config import load_config

# Load configuration
config = load_config("configs/base.yaml")

# Create pipeline
pipeline = RAGPipeline(config=config)

# Load FAISS index (if you have one built)
pipeline.load_index("data/processed/index")

# Query
result = pipeline.query(
    question="What is BRCA1?",
    question_type="factoid"
)

print(result.answer.answer)
print(f"Citations: {result.answer.citations}")
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and medical disclaimer |
| `/api/v1/health` | GET | Health check with pipeline status |
| `/api/v1/config` | GET | Current configuration |
| `/api/v1/answer` | POST | Full RAG answer with citations |
| `/api/v1/retrieve` | POST | Retrieve chunks only (no generation) |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

### Example: Answer Endpoint

```bash
curl -X POST "http://localhost:8000/api/v1/answer" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is BRCA1?", "question_type": "factoid"}'
```

Response:
```json
{
  "answer": "BRCA1 is a tumor suppressor gene that plays a critical role in DNA repair...",
  "answer_type": "direct",
  "citations": [{"pmid": "32171076", "chunk_id": "32171076_0"}],
  "abstained": false,
  "supported_by_evidence": true,
  "latency": {
    "retrieve_ms": 32.3,
    "rerank_ms": 11.4,
    "generate_ms": 4373.6,
    "total_ms": 5693.4
  }
}
```

---

## 🛠️ CLI Commands Reference

### Core Commands

| Command | Description | Example |
|---------|-------------|---------|
| `biorag --help` | Show all available commands | `biorag --help` |
| `biorag info` | Show current configuration and system info | `biorag info` |
| `biorag serve` | Start FastAPI server | `biorag serve --port 8000` |

### Data Ingestion

| Command | Description | Example |
|---------|-------------|---------|
| `biorag ingest-bioasq` | Load and validate BioASQ dataset | `biorag ingest-bioasq --output data/raw/bioasq` |
| `biorag ingest-pubmedqa` | Load and validate PubMedQA dataset | `biorag ingest-pubmedqa --split pqa_labeled` |
| `biorag build-corpus` | Build corpus from PubMed abstracts | `biorag build-corpus --num-distractors 10000` |

### Indexing

| Command | Description | Example |
|---------|-------------|---------|
| `biorag index-faiss` | Build FAISS index from corpus | `biorag index-faiss --chunk-size 350` |

### Query & Debug

| Command | Description | Example |
|---------|-------------|---------|
| `biorag retrieve` | Retrieve chunks for a query (debugging) | `biorag retrieve "What is BRCA1?" --k 10` |

### Evaluation

| Command | Description | Example |
|---------|-------------|---------|
| `biorag eval` | Run evaluation on golden suite | `biorag eval --dataset pubmedqa --limit 100` |

#### Evaluation Options

```bash
# Run on PubMedQA with 100 questions
biorag eval --dataset pubmedqa --limit 100

# Run on BioASQ (requires trust_remote_code)
biorag eval --dataset bioasq --limit 50

# Custom output directory
biorag eval --dataset pubmedqa --output-dir runs/my_eval
```

### Parameter Sweeps

| Command | Description | Example |
|---------|-------------|---------|
| `biorag sweep` | Run parameter sweep with RapidFire AI | `biorag sweep --config configs/sweeps/full_sweep.yaml` |

#### Sweep Options

```bash
# Run sweep with specific config
biorag sweep --config configs/sweeps/retriever_sweep.yaml

# Sequential execution (no parallelization)
biorag sweep --config configs/sweeps/chunking_sweep.yaml --sequential

# Custom output directory
biorag sweep --config configs/sweeps/full_sweep.yaml --output-dir runs/my_sweep
```

#### Available Sweep Configs

| Config | Parameters Swept |
|--------|------------------|
| `chunking_sweep.yaml` | `chunk_size`, `chunk_overlap` |
| `retriever_sweep.yaml` | `mode`, `k`, `fetch_k`, `lambda_mult` |
| `reranker_sweep.yaml` | `model`, `top_n`, `final_k` |
| `prompt_sweep.yaml` | Prompt template variants |
| `full_sweep.yaml` | Combined sweep of all parameters |
| `quick_test.yaml` | Quick 2-config test sweep |
| `quick_test_pubmedqa.yaml` | Quick test with PubMedQA |

---

## 💰 Cost & Latency Report

### API Cost Estimates

| Component | Model | Cost per 1M Tokens |
|-----------|-------|-------------------|
| Embeddings | text-embedding-3-large | $0.13 |
| Generation | gpt-4o-mini | $0.15 input / $0.60 output |

#### Typical Per-Query Costs

| Stage | Tokens | Cost |
|-------|--------|------|
| Embedding (query) | ~50 | ~$0.000007 |
| Generation (input) | ~2,000 | ~$0.0003 |
| Generation (output) | ~350 | ~$0.00021 |
| **Total per query** | ~2,400 | **~$0.0005** |

#### Evaluation Run Costs

| Questions | Estimated Cost | Notes |
|-----------|---------------|-------|
| 100 | ~$0.05 | Quick test |
| 500 | ~$0.25 | Standard eval |
| 1,000 | ~$0.50 | Full eval |

### Latency Breakdown

Typical latencies on different hardware:

#### GPU (RTX 3080/T4)

| Stage | Latency | % of Total |
|-------|---------|------------|
| Retrieval (MMR, k=10) | 30-60ms | 1% |
| Reranking (cross-encoder) | 10-20ms | <1% |
| Generation (gpt-4o-mini) | 3-5s | 98% |
| **Total** | **3-6 seconds** | 100% |

#### CPU Only

| Stage | Latency | % of Total |
|-------|---------|------------|
| Retrieval (MMR, k=10) | 50-100ms | 1% |
| Reranking (cross-encoder) | 200-500ms | 5-10% |
| Generation (gpt-4o-mini) | 3-5s | 85-95% |
| **Total** | **4-6 seconds** | 100% |

### Cost Control Features

```yaml
# In configs/base.yaml
cost:
  max_questions: 100        # Hard limit on examples
  max_total_tokens: 500000  # Token budget
  max_usd: 1.00             # Dollar limit
  on_budget_exceeded: fail-fast  # or "skip"
```

### Caching

LLM outputs are cached to avoid re-paying for identical queries:
- Cache location: `data/cache/llm_cache.db`
- Cache key: `hash(model + prompt + params)`
- Typical cache hit rate in sweeps: 60-80%

---

## 📁 Project Structure

```
rag-optimization-biomed-chatbot/
├── configs/                 # Configuration files
│   ├── base.yaml           # Default configuration
│   ├── prompts/            # Prompt templates
│   │   ├── cite_and_abstain_v1.txt
│   │   └── cite_and_abstain_v2.txt
│   └── sweeps/             # Sweep configurations
│       ├── chunking_sweep.yaml
│       ├── retriever_sweep.yaml
│       ├── reranker_sweep.yaml
│       ├── prompt_sweep.yaml
│       └── full_sweep.yaml
├── data/                   # Data directory (gitignored)
│   ├── raw/               # Raw datasets
│   ├── processed/         # Processed corpus & index
│   ├── golden/            # Golden test suites
│   └── cache/             # LLM & embedding cache
├── demo/                   # Gradio demo
│   ├── app.py             # Main demo with side-by-side comparison
│   ├── requirements.txt   # HuggingFace Spaces dependencies
│   └── README.md          # Spaces documentation
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_failure_analysis.ipynb
├── runs/                   # Experiment outputs
│   └── leaderboard.csv    # Sweep results ranking
├── src/biorag/            # Main package
│   ├── api/               # FastAPI backend
│   ├── chunking/          # Text chunking
│   ├── cli/               # CLI commands
│   ├── data/              # Data loaders
│   ├── embeddings/        # Embedding providers
│   ├── eval/              # Evaluation harness
│   ├── experiments/       # Experiment runner & sweeps
│   ├── generate/          # LLM generation
│   ├── indexing/          # FAISS indexing
│   ├── pipeline/          # RAG pipeline
│   ├── rerank/            # Reranking
│   ├── retrieve/          # Retrieval
│   ├── schemas/           # Pydantic models
│   └── utils/             # Utilities
├── tests/                  # Test suite (383 tests)
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── .env                    # Environment variables (gitignored)
├── env.example            # Environment template
├── pyproject.toml         # Project configuration
├── implementation-plan.md # Implementation roadmap
├── SPEC.md                # Technical specification
└── README.md              # This file
```

---

## ⚙️ Configuration

The default configuration is in `configs/base.yaml`:

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 350

embeddings:
  provider: openai  # or "local" for development
  model: text-embedding-3-large

chunking:
  type: recursive
  chunk_size: 350
  chunk_overlap: 40

retrieval:
  mode: mmr  # similarity, mmr, similarity_score_threshold
  k: 10
  fetch_k: 50
  lambda_mult: 0.5

rerank:
  enabled: true
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  final_k: 8
```

### Using Local Embeddings (No API Cost)

For development, you can use local embeddings to avoid OpenAI API costs:

```yaml
# In configs/base.yaml or custom config
embeddings:
  provider: local
  model: all-MiniLM-L6-v2
```

---

## 🔧 Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check src/

# Run formatting
ruff format src/

# Run type checking
mypy src/
```

### Run Tests with Coverage

```bash
python -m pytest tests/ --cov=src/biorag --cov-report=html
# Open htmlcov/index.html in browser
```

---

## 📚 Documentation

- [SPEC.md](SPEC.md) - Full technical specification
- [implementation-plan.md](implementation-plan.md) - Implementation roadmap
- [Failure Analysis Notebook](notebooks/02_failure_analysis.ipynb) - Common failure modes and fixes
- [API Docs](http://localhost:8000/docs) - Swagger UI (when server is running)

---

## 🗺️ Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ Complete | Project foundation |
| 1 | ✅ Complete | Data loaders (BioASQ, PubMedQA) |
| 2 | ✅ Complete | Indexing pipeline (chunking, FAISS) |
| 3 | ✅ Complete | Retrieval and reranking |
| 4 | ✅ Complete | Generation with structured outputs |
| 5 | ✅ Complete | RAG pipeline + API + Demo |
| 6 | ✅ Complete | Evaluation harness (metrics, BioASQ, PubMedQA evaluators) |
| 7 | ✅ Complete | Experiment sweeps (RapidFire AI integration) |
| 8 | ✅ Complete | Side-by-side demo + HuggingFace Spaces deployment |

---

## 🤗 HuggingFace Spaces Deployment

The demo is designed for deployment on HuggingFace Spaces:

### Recommended Configuration

| Setting | Value |
|---------|-------|
| SDK | Gradio |
| Hardware | T4 Small (GPU) |
| Python | 3.11+ |

### Required Secrets

Set these as Repository Secrets in Space settings:

| Secret | Description |
|--------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |

See [demo/README.md](demo/README.md) for detailed deployment instructions.

---

## 📄 License

[MIT License](LICENSE)

---

## 🙏 Acknowledgments

- [BioASQ](http://bioasq.org/) - Biomedical Question Answering benchmark
- [PubMedQA](https://pubmedqa.github.io/) - PubMed-based QA dataset
- [LangChain](https://langchain.com/) - LLM application framework
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [OpenAI](https://openai.com/) - LLM and embedding APIs
- [RapidFire AI](https://rapidfire.ai/) - Hyperparameter optimization
