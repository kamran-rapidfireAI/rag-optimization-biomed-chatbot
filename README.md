# BioRAG Bench

Benchmark-driven biomedical RAG optimization (LangChain + FAISS + OpenAI) with golden tests (BioASQ + PubMedQA).

> ⚠️ **Disclaimer:** This is an educational/research project. Not intended for medical advice.

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS 12+ | Windows 11, Ubuntu 22.04 |
| **CPU** | 8 cores / 16 threads | 12-16 cores / 24-32 threads |
| **RAM** | 32 GB | 64 GB |
| **GPU** | NVIDIA RTX 3060 (12GB VRAM) | RTX 3080/4070 Ti (10-12GB) or RTX 3090 (24GB) |
| **Storage** | 256 GB NVMe SSD | 512 GB - 1 TB NVMe SSD |
| **Network** | 50+ Mbps | 100+ Mbps |

> ⚠️ **GPU Required:** Cross-encoder reranking requires a CUDA-compatible NVIDIA GPU. This is a core component of the pipeline and cannot be skipped.

### Why These Specs?

- **GPU (12GB+ VRAM):** Cross-encoder reranking runs on GPU for acceptable latency (~100-200ms vs 2-5s on CPU). Optional local embeddings also benefit from GPU.
- **RAM (32GB+):** FAISS index, embedding cache, and PyTorch overhead run concurrently.
- **CPU (8+ cores):** RapidFire AI sweeps leverage parallelization (16-24x throughput).
- **Storage (256GB+ NVMe):** Corpus, embeddings cache, FAISS index, and experiment artifacts.

### Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| Python | 3.12.x | Required |
| CUDA | 11.8+ | For GPU support |
| cuDNN | 8.6+ | For PyTorch GPU |
| Git | 2.30+ | Version control |

### API Keys

You will need an OpenAI API key for embeddings and generation:

```bash
# Create .env file from template
cp .env.example .env

# Add your API key
OPENAI_API_KEY=sk-...
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/rapidfireai-repos/rag-optimization-biomed-chatbot.git
cd rag-optimization-biomed-chatbot

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -e .

# Download datasets
python scripts/download_datasets.py

# Build corpus and index
biorag build_corpus
biorag index_faiss

# Run evaluation
biorag eval --golden-suite

# Start demo server
biorag serve
```

---

## Project Structure

See [SPEC.md](SPEC.md) for the full specification, including:

- Functional requirements (data loading, chunking, retrieval, reranking, generation)
- Evaluation harness (BioASQ + PubMedQA metrics)
- RapidFire AI sweep integration
- API and CLI design

---

## License

[Add license information]
