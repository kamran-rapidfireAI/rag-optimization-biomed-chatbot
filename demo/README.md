---
title: BioRAG Bench
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.19.0
app_file: app.py
pinned: false
license: mit
---

# 🧬 BioRAG Bench — Biomedical RAG Demo

A **side-by-side comparison** demo for biomedical question answering using Retrieval-Augmented Generation (RAG).

> ⚠️ **Medical Disclaimer:** This is an educational/research project. It should NOT be used for medical diagnosis, treatment decisions, or as a substitute for professional medical advice.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Side-by-Side Comparison** | Compare Baseline vs Optimized RAG configurations |
| **MMR Retrieval** | Maximal Marginal Relevance for diverse evidence |
| **Cross-Encoder Reranking** | GPU-accelerated semantic reranking |
| **Structured Outputs** | JSON-formatted answers with verified citations |
| **Abstention Logic** | Refuses to answer when evidence is insufficient |

---

## 🔬 Configurations Compared

| Feature | 🔵 Baseline | 🟢 Optimized |
|---------|------------|--------------|
| Retrieval Mode | Similarity | MMR |
| Top-K Documents | 5 | 10 |
| Fetch-K | 20 | 50 |
| Reranking | ❌ Disabled | ✅ Cross-Encoder |
| Final Documents | 5 | 8 |

---

## 🚀 HuggingFace Spaces Deployment

### Recommended Space Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| **SDK** | Gradio | Required |
| **SDK Version** | 4.19.0+ | Tested version |
| **Hardware** | T4 Small (GPU) | Required for cross-encoder reranking |
| **Python** | 3.11+ | 3.12 recommended |

### Hardware Requirements

| Tier | Specs | Cost | Recommendation |
|------|-------|------|----------------|
| CPU Basic (Free) | 2 vCPU, 16 GB RAM | Free | ❌ Not recommended (slow reranking) |
| CPU Upgrade | 8 vCPU, 32 GB RAM | ~$0.03/hr | ⚠️ Usable but slow (~2-5s/query) |
| **T4 Small** | T4 GPU, 4 vCPU, 15 GB RAM | ~$0.40/hr | ✅ **Recommended** |
| A10G Small | A10G GPU, 4 vCPU, 15 GB RAM | ~$1.05/hr | ✅ Best performance |

> ⚠️ **GPU Strongly Recommended:** The cross-encoder reranker is 10-20x faster on GPU (~15ms vs ~300ms on CPU).

### Required Secrets

Configure these as **Repository Secrets** in Space settings:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key for embeddings & generation |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BIORAG_CONFIG` | `../configs/base.yaml` | Path to config file |
| `BIORAG_INDEX` | `../data/processed/index` | Path to FAISS index |

---

## 🖥️ Local Development

### Quick Start

```bash
# From project root
cd demo

# Install demo-specific dependencies
pip install -r requirements.txt

# Run Gradio app
python app.py

# With options
python app.py --host 0.0.0.0 --port 7860 --share
```

The app will be available at `http://localhost:7860`.

### Using the Main Package

For full functionality, install the main package:

```bash
# From project root
pip install -e .

# Run demo
python demo/app.py
```

---

## 📊 Demo Tabs

### ⚖️ Side-by-Side Comparison

Compare Baseline and Optimized pipelines on the same question:

- **Baseline**: Simple similarity retrieval, no reranking
- **Optimized**: MMR retrieval + cross-encoder reranking

Each panel shows:
- Generated answer with citations
- Retrieved evidence chunks with scores
- Latency breakdown (retrieve/rerank/generate)
- Configuration summary

### 🔍 Single Query

Standard single-pipeline mode for quick queries.

### ℹ️ About

Information about the pipeline architecture and technology stack.

---

## 🏗️ Architecture

```
Question
    ↓
Embedding (OpenAI text-embedding-3-large)
    ↓
FAISS Retrieval (similarity or MMR)
    ↓
Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
    ↓
LLM Generation (GPT-4o-mini)
    ↓
Structured Answer with Citations
```

---

## 📁 Files

| File | Description |
|------|-------------|
| `app.py` | Main Gradio interface with side-by-side comparison |
| `requirements.txt` | Dependencies for HuggingFace Spaces |
| `README.md` | This file (displayed on Space page) |

---

## ⚡ Performance

Typical latencies on T4 GPU:

| Stage | Baseline | Optimized |
|-------|----------|-----------|
| Retrieval | 30-50ms | 40-70ms |
| Reranking | N/A | 10-20ms |
| Generation | 3-5s | 3-5s |
| **Total** | **3-5s** | **3-5s** |

---

## 📚 Datasets

- **BioASQ**: Biomedical semantic QA benchmark
- **PubMedQA**: PubMed-based question answering

---

## 🔗 Links

- [GitHub Repository](https://github.com/yourusername/biorag-bench)
- [Technical Documentation](https://github.com/yourusername/biorag-bench/blob/main/SPEC.md)
- [Implementation Plan](https://github.com/yourusername/biorag-bench/blob/main/implementation-plan.md)

---

## 📄 License

MIT License - See [LICENSE](../LICENSE) for details.
