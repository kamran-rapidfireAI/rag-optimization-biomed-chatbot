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

### Step-by-Step Deployment Guide

Follow these steps to deploy BioRAG Bench to HuggingFace Spaces:

#### Step 1: Prerequisites

Before deploying, ensure you have:

- [ ] A [HuggingFace account](https://huggingface.co/join)
- [ ] An [OpenAI API key](https://platform.openai.com/api-keys)
- [ ] Git installed on your machine
- [ ] The BioRAG Bench repository cloned locally

#### Step 2: Build the Demo Index

The demo requires a pre-built FAISS index. Build it locally first:

```bash
# Navigate to project root
cd /path/to/rag-optimization-biomed-chatbot

# Activate virtual environment
source .venv/bin/activate

# Load environment variables (for OpenAI API key)
source .env

# Build the demo index (uses OpenAI embeddings)
python3 scripts/build_demo_index.py --num-documents 500 --output demo/index
```

This creates:
- `demo/index/index.faiss` — The FAISS vector index
- `demo/index/metadata.db` — Document metadata
- `demo/index/config.json` — Index configuration

#### Step 3: Create a New HuggingFace Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the Space details:
   - **Owner**: Your username or organization
   - **Space name**: `biorag-bench` (or your preferred name)
   - **License**: MIT
   - **SDK**: Select **Gradio**
   - **Hardware**: Select **T4 Small** (GPU recommended)
   - **Visibility**: Public or Private

3. Click **Create Space**

#### Step 4: Configure Secrets

Your Space needs the OpenAI API key to function:

1. Go to your Space's **Settings** tab
2. Scroll to **Repository secrets**
3. Click **New secret**
4. Add the following secret:

| Name | Value |
|------|-------|
| `OPENAI_API_KEY` | `sk-your-openai-api-key-here` |

5. Click **Save**

#### Step 5: Clone and Prepare Files

```bash
# Clone your new Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/biorag-bench
cd biorag-bench

# Copy required files from the project
cp -r /path/to/rag-optimization-biomed-chatbot/demo/* .
cp -r /path/to/rag-optimization-biomed-chatbot/src/biorag ./biorag
cp -r /path/to/rag-optimization-biomed-chatbot/configs ./configs
cp -r /path/to/rag-optimization-biomed-chatbot/prompts ./prompts

# Verify theme.py is included (required for styling)
ls -la app.py theme.py requirements.txt
```

#### Step 6: Update app.py Paths

Edit `app.py` to adjust paths for the Spaces environment:

```python
# Change these lines near the top of app.py:
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR  # In Spaces, everything is in the same directory
CONFIG_PATH = os.environ.get("BIORAG_CONFIG", PROJECT_ROOT / "configs" / "base.yaml")
INDEX_PATH = os.environ.get("BIORAG_INDEX", PROJECT_ROOT / "index")
```

#### Step 7: Verify requirements.txt

Ensure `requirements.txt` includes all dependencies:

```txt
gradio>=4.19.0
openai>=1.0.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
langchain>=0.1.0
langchain-openai>=0.0.5
pydantic>=2.0.0
python-dotenv>=1.0.0
PyYAML>=6.0
rich>=13.0.0
```

#### Step 8: Push to HuggingFace

```bash
# Add all files
git add .

# Commit
git commit -m "Initial deployment of BioRAG Bench demo"

# Push to HuggingFace
git push
```

#### Step 9: Monitor Deployment

1. Go to your Space page: `https://huggingface.co/spaces/YOUR_USERNAME/biorag-bench`
2. Click the **Logs** tab to monitor the build process
3. Wait for the build to complete (typically 3-5 minutes)
4. Once "Running" appears, your demo is live!

#### Step 10: Verify Functionality

1. Open your Space URL
2. Try an example question from the list
3. Verify both Baseline and Optimized pipelines return results
4. Check the Latency panels to confirm GPU acceleration

---

### Alternative: Deploy via HuggingFace CLI

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create Space programmatically
huggingface-cli repo create biorag-bench --type space --space_sdk gradio

# Upload files
huggingface-cli upload YOUR_USERNAME/biorag-bench ./demo --repo-type space
```

---

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
| `BIORAG_CONFIG` | `configs/base.yaml` | Path to config file |
| `BIORAG_INDEX` | `index` | Path to FAISS index |

---

### Troubleshooting

#### Build Fails

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: biorag` | Ensure the `biorag` package is copied to the Space |
| `OPENAI_API_KEY not set` | Add the secret in Space Settings → Repository Secrets |
| `FAISS index not found` | Verify `index/` directory contains `index.faiss` |

#### Runtime Errors

| Error | Solution |
|-------|----------|
| `AssertionError` in FAISS | Index dimension mismatch — rebuild index with same embeddings |
| `Timeout` on queries | Upgrade to GPU hardware tier |
| `RateLimitError` | Check OpenAI API usage limits |

#### Performance Issues

| Issue | Solution |
|-------|----------|
| Slow reranking (~300ms+) | Use GPU hardware (T4 or A10G) |
| High latency on first query | Normal — models are loading |
| Abstention on all queries | Index may be too small or domain-specific |

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

## 🏗️ RAG Pipeline Architecture

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

## 🎨 Theme Architecture (Design Tokens)

The demo uses a **3-layer design token architecture** for consistent, maintainable styling.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                       theme.py                                   │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 1: Primitive Tokens (ColorPrimitives)                     │
│  ─────────────────────────────────────────────                   │
│  Raw color values: gray_900, blue_400, green_500, etc.           │
│                                                                  │
│  LAYER 2: Semantic Tokens (SemanticTokens)                       │
│  ─────────────────────────────────────────                       │
│  Purpose-based: text_primary, bg_surface, accent_primary         │
│                                                                  │
│  LAYER 3: Component Tokens (ComponentTokens)                     │
│  ────────────────────────────────────────────                    │
│  Component-specific: button_primary_bg, code_text, table_text    │
├──────────────────────────────────────────────────────────────────┤
│  OUTPUT GENERATORS                                               │
│  ─────────────────                                               │
│  • to_css()          → Complete CSS with variables               │
│  • to_gradio_theme() → Gradio theme configuration                │
└──────────────────────────────────────────────────────────────────┘
```

### Token Layers Explained

| Layer | Class | Purpose | Example |
|-------|-------|---------|---------|
| **Primitives** | `ColorPrimitives` | Raw color palette | `blue_400 = "#58a6ff"` |
| **Semantic** | `SemanticTokens` | Purpose-based mapping | `text_primary → gray_100` |
| **Component** | `ComponentTokens` | Component-specific | `button_primary_bg → accent_primary` |

### Easy Customization

All theme customization is done in **`demo/theme.py`**. Here are common customization examples:

#### Change Primary Accent Color (Blue → Purple)

```python
# In theme.py, modify SemanticTokens class:
@property
def accent_primary(self) -> str:
    return self.primitives.purple_400  # Was: blue_400
```

#### Change Background to Lighter Theme

```python
# In theme.py, modify ColorPrimitives class:
@dataclass(frozen=True)
class ColorPrimitives:
    gray_900: str = "#1a1a2e"   # Lighter dark background
    gray_850: str = "#25253a"   # Adjusted surface
    # ... other colors
```

#### Add a New Accent Color

```python
# 1. Add to ColorPrimitives
@dataclass(frozen=True)
class ColorPrimitives:
    # ... existing colors ...
    teal_500: str = "#14b8a6"
    teal_400: str = "#2dd4bf"

# 2. Create semantic token
@property
def accent_teal(self) -> str:
    return self.primitives.teal_400

# 3. Use in components
@property
def pipeline_optimized_accent(self) -> str:
    return self.semantic.accent_teal  # Was: accent_success
```

#### Customize Button Appearance

```python
# In ComponentTokens class:
@property
def button_primary_bg(self) -> str:
    return self.semantic.primitives.green_500  # Green buttons

@property
def button_primary_bg_hover(self) -> str:
    return self.semantic.primitives.green_400
```

### Color Palette Reference

The default theme uses a **GitHub Dark**-inspired palette:

| Token | Value | Preview |
|-------|-------|---------|
| `gray_900` | `#0d1117` | ![#0d1117](https://via.placeholder.com/20/0d1117/0d1117) App background |
| `gray_850` | `#161b22` | ![#161b22](https://via.placeholder.com/20/161b22/161b22) Surface |
| `gray_800` | `#21262d` | ![#21262d](https://via.placeholder.com/20/21262d/21262d) Elevated |
| `gray_100` | `#e6edf3` | ![#e6edf3](https://via.placeholder.com/20/e6edf3/e6edf3) Primary text |
| `blue_400` | `#58a6ff` | ![#58a6ff](https://via.placeholder.com/20/58a6ff/58a6ff) Primary accent |
| `green_500` | `#3fb950` | ![#3fb950](https://via.placeholder.com/20/3fb950/3fb950) Success |
| `red_500` | `#f85149` | ![#f85149](https://via.placeholder.com/20/f85149/f85149) Error |
| `purple_400` | `#a371f7` | ![#a371f7](https://via.placeholder.com/20/a371f7/a371f7) Secondary accent |

### Using the Theme Programmatically

```python
from demo.theme import BioRAGTheme

# Create theme instance
theme = BioRAGTheme()

# Access any token
print(theme.primitives.blue_400)        # "#58a6ff"
print(theme.semantic.text_primary)      # "#e6edf3"
print(theme.components.button_primary_bg)  # "#58a6ff"

# Generate outputs
css = theme.to_css()                    # Complete CSS string
gradio_theme = theme.to_gradio_theme()  # Gradio theme object
```

### Benefits of This Architecture

| Benefit | Description |
|---------|-------------|
| **Single Source of Truth** | All colors defined in one file |
| **Semantic Naming** | `text_primary` is clearer than `#e6edf3` |
| **Easy Maintenance** | Change one token, update everywhere |
| **Type Safety** | Dataclasses with IDE autocomplete |
| **Separation of Concerns** | Primitives → Semantics → Components |
| **No `!important` Spam** | Clean CSS with proper specificity |

---

## 📁 Files

| File | Description |
|------|-------------|
| `app.py` | Main Gradio interface with side-by-side comparison |
| `theme.py` | Design token architecture for colors and styling |
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
