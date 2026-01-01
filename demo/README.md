# BioRAG Bench — Gradio Demo

This directory contains the Gradio demo application for deployment on HuggingFace Spaces.

---

## HuggingFace Spaces Deployment

### Recommended Space Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| **SDK** | Gradio | Required |
| **Hardware** | T4 Small (GPU) | Required for cross-encoder reranking |
| **Python** | 3.12 | Match development environment |

### Hardware Tiers

| Tier | Specs | Cost | Recommendation |
|------|-------|------|----------------|
| CPU Basic (Free) | 2 vCPU, 16 GB RAM | Free | ❌ Not recommended (no reranking) |
| CPU Upgrade | 8 vCPU, 32 GB RAM | ~$0.03/hr | ⚠️ Slow reranking (~2-5s/query) |
| **T4 Small** | T4 GPU (16GB), 4 vCPU, 15 GB RAM | ~$0.40/hr | ✅ **Recommended** |
| A10G Small | A10G GPU (24GB), 4 vCPU, 15 GB RAM | ~$1.05/hr | ✅ Best performance |

> ⚠️ **GPU Required:** The cross-encoder reranker is a core component. Without GPU, reranking latency increases from ~100ms to ~2-5 seconds per query, significantly degrading user experience.

### Alternative: Self-Hosted Deployment

If deploying outside HuggingFace Spaces (e.g., AWS, GCP, Azure):

#### Minimum Specs

| Component | Specification |
|-----------|---------------|
| **CPU** | 4 vCPU |
| **RAM** | 16 GB |
| **GPU** | T4 (16GB VRAM) or equivalent |
| **Storage** | 50 GB SSD |

#### Recommended Specs

| Component | Specification |
|-----------|---------------|
| **CPU** | 8 vCPU |
| **RAM** | 32 GB |
| **GPU** | A10G (24GB VRAM) or RTX A4000 |
| **Storage** | 100 GB SSD |

#### Cloud Provider Examples

| Provider | Instance Type | vCPU | RAM | GPU | Approx. Cost |
|----------|---------------|------|-----|-----|--------------|
| AWS | g4dn.xlarge | 4 | 16 GB | T4 (16GB) | ~$0.53/hr |
| AWS | g5.xlarge | 4 | 16 GB | A10G (24GB) | ~$1.00/hr |
| GCP | n1-standard-4 + T4 | 4 | 15 GB | T4 (16GB) | ~$0.45/hr |
| Azure | NC4as T4 v3 | 4 | 28 GB | T4 (16GB) | ~$0.53/hr |

---

## Demo Features

The Gradio demo provides a **side-by-side comparison** view:

| Left Panel | Right Panel |
|------------|-------------|
| Baseline RAG configuration | Optimized RAG configuration |

### Each Response Shows

- Answer with inline citations
- Retrieved chunks with scores (before/after rerank)
- Latency breakdown (retrieve / rerank / generate)
- Configuration summary (chunking, retriever, reranker settings)

---

## Local Development

To run the demo locally:

```bash
# From project root
cd demo

# Install demo-specific dependencies
pip install -r requirements.txt

# Run Gradio app
python app.py
```

The app will be available at `http://localhost:7860`.

---

## Environment Variables

The demo requires the following environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key for generation |
| `FAISS_INDEX_PATH` | ❌ | Path to FAISS index (default: `../data/processed/faiss_index`) |
| `CONFIG_PATH` | ❌ | Path to config file (default: `../configs/base.yaml`) |

For HuggingFace Spaces, set these as **Repository Secrets** in the Space settings.

---

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Gradio interface |
| `requirements.txt` | Minimal dependencies for Spaces |
| `README.md` | This file (also displayed on the Space page) |
