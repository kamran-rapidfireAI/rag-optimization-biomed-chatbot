"""
BioRAG Bench Gradio Demo

A biomedical question answering system using RAG (Retrieval-Augmented Generation).
This demo provides a user-friendly interface for asking biomedical questions
and receiving evidence-backed answers.

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


# Medical disclaimer HTML
DISCLAIMER_HTML = """
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
            border-left: 4px solid #e94560; 
            padding: 16px 20px; 
            border-radius: 8px; 
            margin-bottom: 24px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);">
    <div style="display: flex; align-items: center; margin-bottom: 8px;">
        <span style="font-size: 24px; margin-right: 10px;">⚠️</span>
        <strong style="color: #e94560; font-size: 16px; letter-spacing: 0.5px;">MEDICAL DISCLAIMER</strong>
    </div>
    <p style="color: #a0aec0; margin: 0; line-height: 1.6; font-size: 14px;">
        This system is for <strong style="color: #edf2f7;">research and educational purposes only</strong>. 
        It should <strong style="color: #e94560;">NOT</strong> be used for medical diagnosis, 
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
        return f"[{idx}] PMID: {pmid} — \"{quote[:100]}...\""
    return f"[{idx}] PMID: {pmid}"


def format_chunk(chunk: dict[str, Any], idx: int) -> str:
    """Format a retrieved chunk for display."""
    pmid = chunk.get("pmid", "Unknown")
    text = chunk.get("text", "")[:300]
    score = chunk.get("rerank_score") or chunk.get("score", 0)
    rank = chunk.get("rerank_rank") or chunk.get("rank", idx + 1)
    
    return f"""
**#{rank}** | PMID: `{pmid}` | Score: `{score:.4f}`

{text}...
"""


def format_latency(latency: dict[str, float]) -> str:
    """Format latency breakdown for display."""
    return f"""
| Stage | Time |
|-------|------|
| Retrieval | {latency.get('retrieve_ms', 0):.1f} ms |
| Reranking | {latency.get('rerank_ms', 0):.1f} ms |
| Generation | {latency.get('generate_ms', 0):.1f} ms |
| **Total** | **{latency.get('total_ms', 0):.1f} ms** |
"""


class BioRAGDemo:
    """Gradio demo for BioRAG Bench."""

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
        self.config_path = config_path
        self.index_path = index_path
        self._pipeline: Any = None

    @property
    def pipeline(self) -> Any:
        """Lazy-load the RAG pipeline."""
        if self._pipeline is None:
            from biorag.pipeline.rag import RAGPipeline
            from biorag.schemas.config import load_config

            config = load_config(self.config_path)
            self._pipeline = RAGPipeline(config=config)

            # Load FAISS index if available
            if self.index_path and Path(self.index_path).exists():
                self._pipeline.load_index(self.index_path)

        return self._pipeline

    def answer_question(
        self,
        question: str,
        question_type: str,
    ) -> tuple[str, str, str, str]:
        """
        Answer a biomedical question.

        Args:
            question: The question to answer
            question_type: Type of question (auto, yesno, factoid, list, summary)

        Returns:
            Tuple of (answer, citations, chunks, latency)
        """
        if not question.strip():
            return (
                "Please enter a question.",
                "",
                "",
                "",
            )

        try:
            # Map question type
            q_type = None if question_type == "auto" else question_type

            # Run the pipeline
            result = self.pipeline.query(
                question=question,
                question_type=q_type,
            )

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
                citations_text = "### Citations\n\n"
                for i, cit in enumerate(answer.citations, 1):
                    citations_text += format_citation(cit.model_dump(), i) + "\n\n"
            else:
                citations_text = "*No citations provided*"

            # Format retrieved chunks
            chunks_text = "### Retrieved Evidence\n\n"
            for i, chunk in enumerate(result.reranked_chunks[:5]):
                chunks_text += format_chunk(chunk.model_dump(), i) + "\n---\n"

            # Format latency
            latency_text = format_latency(result.latency.to_dict())

            return answer_text, citations_text, chunks_text, latency_text

        except ValueError as e:
            return (
                f"⚠️ **Error**: Pipeline not ready - {e}",
                "",
                "",
                "",
            )
        except Exception as e:
            return (
                f"⚠️ **Error**: {e}",
                "",
                "",
                "",
            )

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        
        # Custom CSS with modern, distinctive styling
        custom_css = """
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        .gradio-container {
            font-family: 'Space Grotesk', sans-serif !important;
            background: linear-gradient(180deg, #0f0f23 0%, #1a1a35 50%, #0f0f23 100%) !important;
            min-height: 100vh;
        }
        
        .main-header {
            text-align: center;
            padding: 32px 0;
            background: linear-gradient(135deg, rgba(233, 69, 96, 0.1) 0%, rgba(52, 152, 219, 0.1) 100%);
            border-radius: 16px;
            margin-bottom: 24px;
            border: 1px solid rgba(233, 69, 96, 0.2);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #e94560 0%, #3498db 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            letter-spacing: -0.5px;
        }
        
        .main-header p {
            color: #a0aec0;
            font-size: 1.1rem;
            margin-top: 8px;
        }
        
        .gr-button-primary {
            background: linear-gradient(135deg, #e94560 0%, #c73e54 100%) !important;
            border: none !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            transition: all 0.3s ease !important;
        }
        
        .gr-button-primary:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(233, 69, 96, 0.4) !important;
        }
        
        .gr-textbox {
            font-family: 'Space Grotesk', sans-serif !important;
        }
        
        .gr-markdown {
            font-family: 'Space Grotesk', sans-serif !important;
        }
        
        .gr-markdown code {
            font-family: 'JetBrains Mono', monospace !important;
            background: rgba(233, 69, 96, 0.15) !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            color: #e94560 !important;
        }
        
        .example-questions {
            background: rgba(52, 152, 219, 0.1);
            border: 1px solid rgba(52, 152, 219, 0.2);
            border-radius: 12px;
            padding: 16px;
            margin-top: 16px;
        }
        """

        with gr.Blocks(
            title="BioRAG Bench",
            theme=gr.themes.Base(
                primary_hue="rose",
                secondary_hue="blue",
                neutral_hue="slate",
                font=("Space Grotesk", "sans-serif"),
            ).set(
                body_background_fill="#0f0f23",
                body_background_fill_dark="#0f0f23",
                block_background_fill="#1a1a35",
                block_background_fill_dark="#1a1a35",
                block_border_color="#2d2d4a",
                block_label_text_color="#a0aec0",
                block_title_text_color="#edf2f7",
                input_background_fill="#252545",
                input_background_fill_dark="#252545",
            ),
            css=custom_css,
        ) as demo:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🧬 BioRAG Bench</h1>
                <p>Biomedical Question Answering with Retrieval-Augmented Generation</p>
            </div>
            """)

            # Medical disclaimer
            gr.HTML(DISCLAIMER_HTML)

            with gr.Row():
                with gr.Column(scale=2):
                    # Question input
                    question_input = gr.Textbox(
                        label="Ask a Biomedical Question",
                        placeholder="e.g., What is the role of BRCA1 in breast cancer?",
                        lines=3,
                        max_lines=5,
                    )

                    with gr.Row():
                        question_type = gr.Dropdown(
                            choices=["auto", "yesno", "factoid", "list", "summary"],
                            value="auto",
                            label="Question Type",
                            info="Select the type of question or let the system detect it",
                        )

                        submit_btn = gr.Button(
                            "🔍 Get Answer",
                            variant="primary",
                            size="lg",
                        )

                    # Example questions
                    gr.Examples(
                        examples=[
                            ["What is the mechanism of action of metformin?", "factoid"],
                            ["Is aspirin effective for preventing heart attacks?", "yesno"],
                            ["What are the symptoms of COVID-19?", "list"],
                            ["Explain the role of p53 in cancer development.", "summary"],
                        ],
                        inputs=[question_input, question_type],
                        label="Example Questions",
                    )

                with gr.Column(scale=3):
                    # Answer output
                    answer_output = gr.Markdown(
                        label="Answer",
                        value="*Enter a question and click 'Get Answer' to see results*",
                    )

            with gr.Row():
                with gr.Column():
                    citations_output = gr.Markdown(
                        label="Citations",
                        value="",
                    )

                with gr.Column():
                    latency_output = gr.Markdown(
                        label="Performance",
                        value="",
                    )

            with gr.Accordion("📚 Retrieved Evidence", open=False):
                chunks_output = gr.Markdown(
                    value="",
                )

            # Config info
            with gr.Accordion("⚙️ Configuration", open=False):
                try:
                    config_summary = self.pipeline.get_config_summary()
                    config_md = f"""
| Setting | Value |
|---------|-------|
| LLM | `{config_summary['llm']['model']}` |
| Embeddings | `{config_summary['embeddings']['model']}` |
| Retrieval | `{config_summary['retrieval']['mode']}` (k={config_summary['retrieval']['k']}) |
| Rerank | `{'enabled' if config_summary['rerank']['enabled'] else 'disabled'}` |
"""
                    gr.Markdown(config_md)
                except Exception:
                    gr.Markdown("*Configuration will be shown when pipeline is initialized*")

            # Footer
            gr.HTML("""
            <div style="text-align: center; padding: 24px; margin-top: 32px; 
                        border-top: 1px solid #2d2d4a; color: #718096;">
                <p style="margin: 0; font-size: 14px;">
                    Built with 🔬 BioRAG Bench | 
                    <a href="https://github.com/yourusername/biorag-bench" 
                       style="color: #e94560; text-decoration: none;">GitHub</a>
                </p>
            </div>
            """)

            # Connect events
            submit_btn.click(
                fn=self.answer_question,
                inputs=[question_input, question_type],
                outputs=[answer_output, citations_output, chunks_output, latency_output],
            )

            question_input.submit(
                fn=self.answer_question,
                inputs=[question_input, question_type],
                outputs=[answer_output, citations_output, chunks_output, latency_output],
            )

        return demo


def create_demo(
    config_path: str | Path | None = None,
    index_path: str | Path | None = None,
) -> gr.Blocks:
    """
    Create the Gradio demo interface.

    Args:
        config_path: Path to configuration file
        index_path: Path to FAISS index directory

    Returns:
        Gradio Blocks interface
    """
    demo = BioRAGDemo(config_path=config_path, index_path=index_path)
    return demo.create_interface()


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
    config_path = args.config or CONFIG_PATH
    index_path = args.index or INDEX_PATH

    print(f"🧬 Starting BioRAG Bench Demo")
    print(f"   Config: {config_path}")
    print(f"   Index: {index_path}")
    print(f"   URL: http://{args.host}:{args.port}")
    print()

    demo = create_demo(config_path=config_path, index_path=index_path)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()





