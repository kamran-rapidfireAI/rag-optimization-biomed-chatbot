"""
BioRAG Bench Gradio Demo - Side-by-Side Comparison

A biomedical question answering system using RAG (Retrieval-Augmented Generation).
This demo provides side-by-side comparison of Baseline vs Optimized configurations.

⚠️ MEDICAL DISCLAIMER: This system is for research and educational purposes only.
It should NOT be used for medical diagnosis, treatment decisions, or as a substitute
for professional medical advice.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import gradio as gr

# Configure paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = os.environ.get("BIORAG_CONFIG", PROJECT_ROOT / "configs" / "base.yaml")
INDEX_PATH = os.environ.get("BIORAG_INDEX", PROJECT_ROOT / "data" / "processed" / "index")

# Default configurations for side-by-side comparison
BASELINE_CONFIG = {
    "retrieval": {"mode": "similarity", "k": 5, "fetch_k": 20},
    "rerank": {"enabled": False},
}

OPTIMIZED_CONFIG = {
    "retrieval": {"mode": "mmr", "k": 10, "fetch_k": 50, "lambda_mult": 0.5},
    "rerank": {"enabled": True, "model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "final_k": 8},
}


# Medical disclaimer HTML with cyber-medical aesthetic
DISCLAIMER_HTML = """
<div style="background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); 
            border: 1px solid #30363d;
            border-left: 4px solid #f85149; 
            padding: 20px 24px; 
            border-radius: 12px; 
            margin-bottom: 28px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.05);">
    <div style="display: flex; align-items: center; margin-bottom: 12px;">
        <span style="font-size: 28px; margin-right: 12px; filter: drop-shadow(0 0 8px rgba(248, 81, 73, 0.5));">⚠️</span>
        <strong style="color: #f85149; font-size: 15px; letter-spacing: 1.5px; text-transform: uppercase; font-weight: 600;">Medical Disclaimer</strong>
    </div>
    <p style="color: #8b949e; margin: 0; line-height: 1.7; font-size: 14px;">
        This system is for <strong style="color: #c9d1d9;">research and educational purposes only</strong>. 
        It should <strong style="color: #f85149;">NOT</strong> be used for medical diagnosis, 
        treatment decisions, or as a substitute for professional medical advice. 
        Always consult qualified healthcare providers for medical questions.
    </p>
</div>
"""


def format_citation(citation: dict[str, Any], idx: int) -> str:
    """Format a single citation for display."""
    pmid = citation.get("pmid", "Unknown")
    quote = citation.get("quote", "")
    
    if quote:
        return f"**[{idx}]** `PMID:{pmid}` — \"{quote[:120]}...\""
    return f"**[{idx}]** `PMID:{pmid}`"


def format_chunk(chunk: dict[str, Any], idx: int, show_rerank: bool = True) -> str:
    """Format a retrieved chunk for display."""
    pmid = chunk.get("pmid", "Unknown")
    text = chunk.get("text", "")[:250]
    
    if show_rerank and chunk.get("rerank_score") is not None:
        score = chunk.get("rerank_score", 0)
        rank = chunk.get("rerank_rank", idx + 1)
        score_label = "Rerank"
    else:
        score = chunk.get("score", 0)
        rank = chunk.get("rank", idx + 1)
        score_label = "Score"
    
    return f"""
**#{rank}** • `PMID:{pmid}` • {score_label}: **{score:.4f}**

> {text}...
"""


def format_latency(latency: dict[str, float], show_rerank: bool = True) -> str:
    """Format latency breakdown for display."""
    rows = f"| Retrieval | {latency.get('retrieve_ms', 0):.1f} ms |\n"
    if show_rerank and latency.get('rerank_ms', 0) > 0:
        rows += f"| Reranking | {latency.get('rerank_ms', 0):.1f} ms |\n"
    rows += f"| Generation | {latency.get('generate_ms', 0):.1f} ms |\n"
    rows += f"| **Total** | **{latency.get('total_ms', 0):.1f} ms** |"
    
    return f"""
| Stage | Time |
|-------|------|
{rows}
"""


def format_config_summary(config: dict[str, Any], label: str) -> str:
    """Format configuration summary for display."""
    retrieval = config.get("retrieval", {})
    rerank = config.get("rerank", {})
    
    return f"""
### {label} Configuration

| Setting | Value |
|---------|-------|
| Retrieval Mode | `{retrieval.get('mode', 'similarity')}` |
| Top-K | `{retrieval.get('k', 5)}` |
| Fetch-K | `{retrieval.get('fetch_k', 20)}` |
| Reranking | `{'✅ Enabled' if rerank.get('enabled', False) else '❌ Disabled'}` |
| Rerank Model | `{rerank.get('model', 'N/A') if rerank.get('enabled') else 'N/A'}` |
| Final-K | `{rerank.get('final_k', 'N/A') if rerank.get('enabled') else 'N/A'}` |
"""


class BioRAGDemo:
    """Gradio demo for BioRAG Bench with side-by-side comparison."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        index_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the demo.

        Args:
            config_path: Path to configuration file
            index_path: Path to FAISS index directory
        """
        self.config_path = config_path or CONFIG_PATH
        self.index_path = index_path or INDEX_PATH
        self._baseline_pipeline: Any = None
        self._optimized_pipeline: Any = None
        self._single_pipeline: Any = None

    def _create_pipeline(self, config_overrides: dict[str, Any] | None = None) -> Any:
        """Create a RAG pipeline with optional config overrides."""
        from biorag.pipeline.rag import RAGPipeline
        from biorag.schemas.config import load_config

        config = load_config(self.config_path)
        
        # Apply overrides if provided
        if config_overrides:
            if "retrieval" in config_overrides:
                for k, v in config_overrides["retrieval"].items():
                    setattr(config.retrieval, k, v)
            if "rerank" in config_overrides:
                for k, v in config_overrides["rerank"].items():
                    setattr(config.rerank, k, v)
        
        pipeline = RAGPipeline(config=config)

        # Load FAISS index if available
        index_path = Path(self.index_path)
        if index_path.exists():
            pipeline.load_index(index_path)

        return pipeline

    @property
    def baseline_pipeline(self) -> Any:
        """Lazy-load the baseline RAG pipeline."""
        if self._baseline_pipeline is None:
            self._baseline_pipeline = self._create_pipeline(BASELINE_CONFIG)
        return self._baseline_pipeline

    @property
    def optimized_pipeline(self) -> Any:
        """Lazy-load the optimized RAG pipeline."""
        if self._optimized_pipeline is None:
            self._optimized_pipeline = self._create_pipeline(OPTIMIZED_CONFIG)
        return self._optimized_pipeline

    @property
    def single_pipeline(self) -> Any:
        """Lazy-load a single pipeline for simple mode."""
        if self._single_pipeline is None:
            self._single_pipeline = self._create_pipeline()
        return self._single_pipeline

    def _process_result(
        self,
        result: Any,
        config: dict[str, Any],
        config_label: str,
    ) -> tuple[str, str, str, str, str]:
        """Process a pipeline result into formatted display strings."""
        # Format answer
        answer = result.answer
        if answer.abstained:
            answer_text = f"⚠️ **Unable to answer**: {answer.abstention_reason or 'Insufficient evidence'}"
        else:
            answer_text = answer.answer
            if answer.label:
                answer_text = f"**{answer.label.upper()}**: {answer_text}"
            if answer.confidence:
                answer_text += f"\n\n*Confidence: {answer.confidence:.2%}*"

        # Format citations
        citations_text = ""
        if answer.citations:
            citations_text = "### 📑 Citations\n\n"
            for i, cit in enumerate(answer.citations, 1):
                citations_text += format_citation(cit.model_dump(), i) + "\n\n"
        else:
            citations_text = "*No citations provided*"

        # Format retrieved chunks
        show_rerank = config.get("rerank", {}).get("enabled", False)
        chunks_text = "### 📚 Retrieved Evidence\n\n"
        chunks = result.reranked_chunks if show_rerank else result.retrieved_chunks
        for i, chunk in enumerate(chunks[:5]):
            chunks_text += format_chunk(chunk.model_dump(), i, show_rerank) + "\n---\n"

        # Format latency
        latency_text = format_latency(result.latency.to_dict(), show_rerank)

        # Format config
        config_text = format_config_summary(config, config_label)

        return answer_text, citations_text, chunks_text, latency_text, config_text

    def answer_question_single(
        self,
        question: str,
        question_type: str,
    ) -> tuple[str, str, str, str]:
        """Answer a biomedical question using the single pipeline."""
        if not question.strip():
            return ("Please enter a question.", "", "", "")

        try:
            q_type = None if question_type == "auto" else question_type
            result = self.single_pipeline.query(question=question, question_type=q_type)
            
            answer_text, citations_text, chunks_text, latency_text, _ = self._process_result(
                result, OPTIMIZED_CONFIG, "Default"
            )
            return answer_text, citations_text, chunks_text, latency_text

        except ValueError as e:
            return (f"⚠️ **Error**: Pipeline not ready - {e}", "", "", "")
        except Exception as e:
            return (f"⚠️ **Error**: {e}", "", "", "")

    def answer_question_comparison(
        self,
        question: str,
        question_type: str,
    ) -> tuple[str, str, str, str, str, str, str, str, str, str]:
        """
        Answer a biomedical question using both pipelines for comparison.

        Returns:
            Tuple of (baseline_answer, baseline_citations, baseline_chunks, baseline_latency, baseline_config,
                     optimized_answer, optimized_citations, optimized_chunks, optimized_latency, optimized_config)
        """
        if not question.strip():
            empty = "Please enter a question."
            return (empty, "", "", "", "", empty, "", "", "", "")

        q_type = None if question_type == "auto" else question_type

        # Run baseline pipeline
        try:
            baseline_result = self.baseline_pipeline.query(question=question, question_type=q_type)
            baseline_outputs = self._process_result(baseline_result, BASELINE_CONFIG, "🔵 Baseline")
        except Exception as e:
            baseline_outputs = (f"⚠️ **Error**: {e}", "", "", "", "")

        # Run optimized pipeline
        try:
            optimized_result = self.optimized_pipeline.query(question=question, question_type=q_type)
            optimized_outputs = self._process_result(optimized_result, OPTIMIZED_CONFIG, "🟢 Optimized")
        except Exception as e:
            optimized_outputs = (f"⚠️ **Error**: {e}", "", "", "", "")

        return (*baseline_outputs, *optimized_outputs)

    def get_theme(self) -> gr.themes.Base:
        """Get the custom theme for the demo."""
        return gr.themes.Base(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
            font=("IBM Plex Sans", "sans-serif"),
            font_mono=("IBM Plex Mono", "monospace"),
        ).set(
            body_background_fill="#0d1117",
            body_background_fill_dark="#0d1117",
            block_background_fill="#161b22",
            block_background_fill_dark="#161b22",
            block_border_color="#30363d",
            block_label_text_color="#8b949e",
            block_title_text_color="#c9d1d9",
            input_background_fill="#21262d",
            input_background_fill_dark="#21262d",
            input_border_color="#30363d",
            button_primary_background_fill="#1f6feb",
            button_primary_text_color="#ffffff",
        )

    def get_css(self) -> str:
        """Get the custom CSS for the demo."""
        return """
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border-color: #30363d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-purple: #a371f7;
            --accent-orange: #d29922;
        }
        
        .gradio-container {
            font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background: 
                radial-gradient(ellipse at 20% 0%, rgba(88, 166, 255, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 100%, rgba(163, 113, 247, 0.08) 0%, transparent 50%),
                linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%) !important;
            min-height: 100vh;
        }
        
        .main-header {
            text-align: center;
            padding: 40px 24px;
            background: linear-gradient(180deg, rgba(88, 166, 255, 0.05) 0%, transparent 100%);
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 32px;
            position: relative;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 200px;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
        }
        
        .main-header h1 {
            font-size: 2.8rem;
            font-weight: 700;
            color: #ffffff;
            margin: 0 0 8px 0;
            letter-spacing: -1px;
            text-shadow: 0 0 40px rgba(88, 166, 255, 0.3);
        }
        
        .main-header .subtitle {
            color: var(--text-secondary);
            font-size: 1.15rem;
            font-weight: 400;
            margin: 0;
        }
        
        .main-header .dna-icon {
            font-size: 3rem;
            display: block;
            margin-bottom: 16px;
            filter: drop-shadow(0 0 20px rgba(88, 166, 255, 0.4));
        }
        
        .comparison-header {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            font-weight: 600;
            font-size: 14px;
            letter-spacing: 0.5px;
        }
        
        .baseline-header {
            background: linear-gradient(135deg, rgba(88, 166, 255, 0.15) 0%, rgba(88, 166, 255, 0.05) 100%);
            border: 1px solid rgba(88, 166, 255, 0.3);
            color: var(--accent-blue);
        }
        
        .optimized-header {
            background: linear-gradient(135deg, rgba(63, 185, 80, 0.15) 0%, rgba(63, 185, 80, 0.05) 100%);
            border: 1px solid rgba(63, 185, 80, 0.3);
            color: var(--accent-green);
        }
        
        .gr-button-primary {
            background: linear-gradient(135deg, var(--accent-blue) 0%, #1f6feb 100%) !important;
            border: none !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 12px rgba(88, 166, 255, 0.25) !important;
        }
        
        .gr-button-primary:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(88, 166, 255, 0.4) !important;
        }
        
        .gr-textbox, .gr-dropdown {
            font-family: 'IBM Plex Sans', sans-serif !important;
        }
        
        .gr-markdown {
            font-family: 'IBM Plex Sans', sans-serif !important;
        }
        
        .gr-markdown code {
            font-family: 'IBM Plex Mono', monospace !important;
            background: rgba(88, 166, 255, 0.1) !important;
            padding: 2px 8px !important;
            border-radius: 4px !important;
            color: var(--accent-blue) !important;
            font-size: 0.9em !important;
        }
        
        .gr-markdown table {
            border-collapse: separate;
            border-spacing: 0;
            width: 100%;
            margin: 12px 0;
        }
        
        .gr-markdown th {
            background: var(--bg-tertiary);
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
            color: var(--text-primary);
            border-bottom: 2px solid var(--border-color);
        }
        
        .gr-markdown td {
            padding: 8px 12px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-secondary);
        }
        
        .gr-markdown blockquote {
            border-left: 3px solid var(--accent-purple);
            background: rgba(163, 113, 247, 0.05);
            padding: 12px 16px;
            margin: 12px 0;
            border-radius: 0 8px 8px 0;
            color: var(--text-secondary);
            font-style: italic;
        }
        
        .tab-nav button {
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-weight: 500 !important;
        }
        
        .footer {
            text-align: center;
            padding: 32px;
            margin-top: 48px;
            border-top: 1px solid var(--border-color);
            background: linear-gradient(180deg, transparent 0%, rgba(88, 166, 255, 0.02) 100%);
        }
        
        .footer p {
            color: var(--text-secondary);
            font-size: 14px;
            margin: 0;
        }
        
        .footer a {
            color: var(--accent-blue);
            text-decoration: none;
            transition: color 0.2s;
        }
        
        .footer a:hover {
            color: var(--accent-purple);
        }
        """

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface with side-by-side comparison."""
        
        # Note: theme and css are passed to launch() in Gradio 6.0+
        with gr.Blocks(title="BioRAG Bench") as demo:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <span class="dna-icon">🧬</span>
                <h1>BioRAG Bench</h1>
                <p class="subtitle">Biomedical Question Answering with Retrieval-Augmented Generation</p>
            </div>
            """)

            # Medical disclaimer
            gr.HTML(DISCLAIMER_HTML)

            # Tabs for Simple vs Comparison mode
            with gr.Tabs() as tabs:
                
                # Tab 1: Side-by-side Comparison
                with gr.TabItem("⚖️ Side-by-Side Comparison", id="comparison"):
                    gr.HTML("""
                    <div style="text-align: center; padding: 16px; color: #8b949e; margin-bottom: 16px;">
                        <p style="margin: 0;">Compare <strong style="color: #58a6ff;">Baseline</strong> (simple retrieval) 
                        vs <strong style="color: #3fb950;">Optimized</strong> (MMR + reranking) configurations</p>
                    </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            comparison_question = gr.Textbox(
                                label="Ask a Biomedical Question",
                                placeholder="e.g., What is the role of BRCA1 in breast cancer?",
                                lines=2,
                                max_lines=4,
                            )
                        with gr.Column(scale=1):
                            comparison_type = gr.Dropdown(
                                choices=["auto", "yesno", "factoid", "list", "summary"],
                                value="auto",
                                label="Question Type",
                            )
                            comparison_btn = gr.Button(
                                "🔬 Compare Pipelines",
                                variant="primary",
                                size="lg",
                            )

                    # Example questions
                    gr.Examples(
                        examples=[
                            ["What is the mechanism of action of metformin in diabetes treatment?", "summary"],
                            ["Is aspirin effective for preventing cardiovascular events?", "yesno"],
                            ["What genes are associated with hereditary breast cancer?", "list"],
                            ["What is the function of the p53 tumor suppressor protein?", "factoid"],
                        ],
                        inputs=[comparison_question, comparison_type],
                        label="💡 Example Questions",
                    )

                    # Side-by-side results
                    with gr.Row(equal_height=True):
                        # Baseline column
                        with gr.Column():
                            gr.HTML('<div class="comparison-header baseline-header">🔵 Baseline Pipeline</div>')
                            
                            baseline_answer = gr.Markdown(
                                label="Answer",
                                value="*Results will appear here...*",
                            )
                            
                            with gr.Accordion("📑 Citations", open=False):
                                baseline_citations = gr.Markdown(value="")
                            
                            with gr.Accordion("📚 Retrieved Evidence", open=False):
                                baseline_chunks = gr.Markdown(value="")
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    with gr.Accordion("⏱️ Latency", open=True):
                                        baseline_latency = gr.Markdown(value="")
                                with gr.Column(scale=1):
                                    with gr.Accordion("⚙️ Config", open=True):
                                        baseline_config = gr.Markdown(value="")

                        # Optimized column
                        with gr.Column():
                            gr.HTML('<div class="comparison-header optimized-header">🟢 Optimized Pipeline</div>')
                            
                            optimized_answer = gr.Markdown(
                                label="Answer",
                                value="*Results will appear here...*",
                            )
                            
                            with gr.Accordion("📑 Citations", open=False):
                                optimized_citations = gr.Markdown(value="")
                            
                            with gr.Accordion("📚 Retrieved Evidence", open=False):
                                optimized_chunks = gr.Markdown(value="")
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    with gr.Accordion("⏱️ Latency", open=True):
                                        optimized_latency = gr.Markdown(value="")
                                with gr.Column(scale=1):
                                    with gr.Accordion("⚙️ Config", open=True):
                                        optimized_config = gr.Markdown(value="")

                    # Connect comparison events
                    comparison_btn.click(
                        fn=self.answer_question_comparison,
                        inputs=[comparison_question, comparison_type],
                        outputs=[
                            baseline_answer, baseline_citations, baseline_chunks, 
                            baseline_latency, baseline_config,
                            optimized_answer, optimized_citations, optimized_chunks,
                            optimized_latency, optimized_config,
                        ],
                    )
                    
                    comparison_question.submit(
                        fn=self.answer_question_comparison,
                        inputs=[comparison_question, comparison_type],
                        outputs=[
                            baseline_answer, baseline_citations, baseline_chunks,
                            baseline_latency, baseline_config,
                            optimized_answer, optimized_citations, optimized_chunks,
                            optimized_latency, optimized_config,
                        ],
                    )

                # Tab 2: Simple single-pipeline mode
                with gr.TabItem("🔍 Single Query", id="simple"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            simple_question = gr.Textbox(
                                label="Ask a Biomedical Question",
                                placeholder="e.g., What is the role of BRCA1 in breast cancer?",
                                lines=3,
                                max_lines=5,
                            )

                            with gr.Row():
                                simple_type = gr.Dropdown(
                                    choices=["auto", "yesno", "factoid", "list", "summary"],
                                    value="auto",
                                    label="Question Type",
                                    info="Select the type of question or let the system detect it",
                                )
                                simple_btn = gr.Button(
                                    "🔍 Get Answer",
                                    variant="primary",
                                    size="lg",
                                )

                            gr.Examples(
                                examples=[
                                    ["What is the mechanism of action of metformin?", "factoid"],
                                    ["Is aspirin effective for preventing heart attacks?", "yesno"],
                                    ["What are the symptoms of COVID-19?", "list"],
                                    ["Explain the role of p53 in cancer development.", "summary"],
                                ],
                                inputs=[simple_question, simple_type],
                                label="💡 Example Questions",
                            )

                        with gr.Column(scale=3):
                            simple_answer = gr.Markdown(
                                label="Answer",
                                value="*Enter a question and click 'Get Answer' to see results*",
                            )

                    with gr.Row():
                        with gr.Column():
                            simple_citations = gr.Markdown(label="Citations", value="")
                        with gr.Column():
                            simple_latency = gr.Markdown(label="Performance", value="")

                    with gr.Accordion("📚 Retrieved Evidence", open=False):
                        simple_chunks = gr.Markdown(value="")

                    # Connect simple mode events
                    simple_btn.click(
                        fn=self.answer_question_single,
                        inputs=[simple_question, simple_type],
                        outputs=[simple_answer, simple_citations, simple_chunks, simple_latency],
                    )
                    
                    simple_question.submit(
                        fn=self.answer_question_single,
                        inputs=[simple_question, simple_type],
                        outputs=[simple_answer, simple_citations, simple_chunks, simple_latency],
                    )

                # Tab 3: About
                with gr.TabItem("ℹ️ About", id="about"):
                    gr.Markdown("""
## 🧬 About BioRAG Bench

**BioRAG Bench** is a benchmark-driven biomedical RAG optimization pipeline. It demonstrates 
how different retrieval and reranking configurations affect answer quality in biomedical 
question answering.

### Pipeline Architecture

```
Question → Embedding → FAISS Retrieval → Cross-Encoder Reranking → LLM Generation → Answer
```

### Configurations Compared

| Feature | 🔵 Baseline | 🟢 Optimized |
|---------|------------|--------------|
| Retrieval Mode | Similarity | MMR (Maximal Marginal Relevance) |
| Top-K Documents | 5 | 10 |
| Fetch-K | 20 | 50 |
| Reranking | ❌ Disabled | ✅ Cross-Encoder |
| Final Documents | 5 | 8 |

### Key Features

- **MMR Retrieval**: Balances relevance with diversity to avoid redundant evidence
- **Cross-Encoder Reranking**: GPU-accelerated semantic reranking for better precision
- **Structured Outputs**: JSON-formatted answers with verified citations
- **Abstention Logic**: Refuses to answer when evidence is insufficient

### Datasets

- **BioASQ**: Biomedical semantic QA benchmark (factoid, list, yes/no, summary)
- **PubMedQA**: PubMed-based question answering dataset

### Technology Stack

| Component | Technology |
|-----------|------------|
| Vector Store | FAISS |
| Embeddings | OpenAI text-embedding-3-large |
| LLM | GPT-4o-mini |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Framework | LangChain |

---

*Built for research and educational purposes. Not for clinical use.*
                    """)

            # Footer
            gr.HTML("""
            <div class="footer">
                <p>
                    Built with 🧬 <strong>BioRAG Bench</strong> • 
                    <a href="https://github.com/yourusername/biorag-bench" target="_blank">GitHub</a> • 
                    <a href="https://huggingface.co/spaces/yourusername/biorag-bench" target="_blank">HuggingFace Spaces</a>
                </p>
            </div>
            """)

        return demo


def create_demo(
    config_path: str | Path | None = None,
    index_path: str | Path | None = None,
) -> tuple[gr.Blocks, "BioRAGDemo"]:
    """
    Create the Gradio demo interface.

    Args:
        config_path: Path to configuration file
        index_path: Path to FAISS index directory

    Returns:
        Tuple of (Gradio Blocks interface, BioRAGDemo instance)
    """
    demo_instance = BioRAGDemo(config_path=config_path, index_path=index_path)
    interface = demo_instance.create_interface()
    return interface, demo_instance


def main() -> None:
    """Main entry point for the Gradio demo."""
    import argparse

    parser = argparse.ArgumentParser(description="BioRAG Bench Gradio Demo")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--index",
        "-i",
        type=str,
        default=None,
        help="Path to FAISS index directory",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind to",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )

    args = parser.parse_args()

    # Use environment variables as fallback
    config_path = args.config or os.environ.get("BIORAG_CONFIG", CONFIG_PATH)
    index_path = args.index or os.environ.get("BIORAG_INDEX", INDEX_PATH)

    print("🧬 Starting BioRAG Bench Demo")
    print(f"   Config: {config_path}")
    print(f"   Index: {index_path}")
    print(f"   URL: http://{args.host}:{args.port}")
    print()

    interface, demo_instance = create_demo(config_path=config_path, index_path=index_path)
    
    # In Gradio 6.0+, theme and css are passed to launch()
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=demo_instance.get_theme(),
        css=demo_instance.get_css(),
    )


if __name__ == "__main__":
    main()
