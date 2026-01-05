# 🧬 BioRAG Bench

Benchmark-driven biomedical RAG optimization pipeline using LangChain, FAISS, and OpenAI. Evaluates retrieval-augmented generation on BioASQ and PubMedQA golden test suites.

> ⚠️ **Medical Disclaimer:** This is an educational/research project. It should NOT be used for medical diagnosis, treatment decisions, or as a substitute for professional medical advice.

---

## ✨ Features

- **End-to-end RAG Pipeline**: Retrieve → Rerank → Generate with structured JSON outputs
- **Biomedical Focus**: Optimized for PubMed literature and medical Q&A
- **Multiple Retrieval Modes**: Similarity, MMR, and threshold-based retrieval
- **GPU-Accelerated Reranking**: Cross-encoder reranking with CUDA support
- **Citation Enforcement**: Every answer includes verifiable PMID citations
- **Abstention Logic**: Model refuses to answer when evidence is insufficient
- **Cost Controls**: Token limits, budget caps, and LLM output caching
- **FastAPI Backend**: Production-ready REST API with OpenAPI docs
- **Gradio Demo**: Interactive web interface for testing

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

- **383 tests passing**
- **73.55% code coverage**

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

## 🛠️ CLI Commands

| Command | Description |
|---------|-------------|
| `biorag info` | Show current configuration |
| `biorag serve` | Start FastAPI server |
| `biorag ingest-bioasq` | Load BioASQ dataset |
| `biorag ingest-pubmedqa` | Load PubMedQA dataset |
| `biorag build-corpus` | Build corpus from PubMed abstracts |
| `biorag index-faiss` | Build FAISS index |
| `biorag retrieve <query>` | Retrieve chunks for a query |
| `biorag eval` | Run evaluation (Phase 6) |
| `biorag sweep` | Run parameter sweep (Phase 7) |

---

## 📁 Project Structure

```
rag-optimization-biomed-chatbot/
├── configs/                 # Configuration files
│   ├── base.yaml           # Default configuration
│   ├── prompts/            # Prompt templates
│   └── sweeps/             # Sweep configurations
├── data/                   # Data directory (gitignored)
│   ├── raw/               # Raw datasets
│   ├── processed/         # Processed corpus & index
│   ├── golden/            # Golden test suites
│   └── cache/             # LLM & embedding cache
├── demo/                   # Gradio demo
│   └── app.py
├── src/biorag/            # Main package
│   ├── api/               # FastAPI backend
│   ├── chunking/          # Text chunking
│   ├── cli/               # CLI commands
│   ├── data/              # Data loaders
│   ├── embeddings/        # Embedding providers
│   ├── eval/              # Evaluation harness
│   ├── experiments/       # Experiment runner
│   ├── generate/          # LLM generation
│   ├── indexing/          # FAISS indexing
│   ├── pipeline/          # RAG pipeline
│   ├── rerank/            # Reranking
│   ├── retrieve/          # Retrieval
│   ├── schemas/           # Pydantic models
│   └── utils/             # Utilities
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── .env                    # Environment variables (gitignored)
├── env.example            # Environment template
├── pyproject.toml         # Project configuration
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

## 📊 Performance

Typical latencies (on RTX 3080):

| Stage | Latency |
|-------|---------|
| Retrieval (MMR, k=10) | 30-60ms |
| Reranking (cross-encoder) | 10-20ms |
| Generation (gpt-4o-mini) | 3-5 seconds |
| **Total** | **3-6 seconds** |

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
| 8 | 🔄 Next | HuggingFace Spaces deployment + Demo enhancement |

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
