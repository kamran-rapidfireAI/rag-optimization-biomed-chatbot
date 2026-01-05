"""Prompt template management for BioRAG Bench."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from biorag.schemas.evaluation import RetrievalResult
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


class PromptTemplate:
    """
    Manages prompt templates for generation.
    
    Loads templates from files and provides rendering with variables.
    """

    def __init__(
        self,
        template_path: str | Path,
        configs_dir: Path | None = None,
    ) -> None:
        """
        Initialize prompt template.

        Args:
            template_path: Path to template file (relative to configs_dir or absolute)
            configs_dir: Base directory for configs (defaults to project configs/)
        """
        if configs_dir is None:
            # Default to project root/configs
            configs_dir = Path(__file__).parent.parent.parent.parent / "configs"
        
        self.configs_dir = Path(configs_dir)
        self.template_path = template_path
        self._template: str | None = None
        self._template_hash: str | None = None

    @property
    def template(self) -> str:
        """Load and cache template content."""
        if self._template is None:
            self._load_template()
        return self._template  # type: ignore

    @property
    def template_hash(self) -> str:
        """Get stable hash of template content."""
        if self._template_hash is None:
            self._template_hash = hashlib.sha256(self.template.encode()).hexdigest()[:16]
        return self._template_hash

    def _load_template(self) -> None:
        """Load template from file."""
        template_file = Path(self.template_path)
        
        # Try absolute path first, then relative to configs_dir
        if template_file.is_absolute() and template_file.exists():
            path = template_file
        else:
            path = self.configs_dir / self.template_path
        
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        
        self._template = path.read_text()
        logger.debug(f"Loaded template from {path}")

    def render(
        self,
        question: str,
        evidence_chunks: list[RetrievalResult] | list[dict[str, Any]],
        question_type: str = "factoid",
        **kwargs: Any,
    ) -> str:
        """
        Render template with variables.

        Args:
            question: The question to answer
            evidence_chunks: List of evidence chunks (RetrievalResult or dict)
            question_type: Type of question (yesno, factoid, list, summary)
            **kwargs: Additional template variables

        Returns:
            Rendered prompt string
        """
        # Format evidence chunks
        evidence_text = self._format_evidence(evidence_chunks)
        
        # Build variables dict
        variables = {
            "question": question,
            "evidence": evidence_text,
            "question_type": question_type,
            **kwargs,
        }
        
        # Render template
        try:
            rendered = self.template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")
        
        return rendered

    def _format_evidence(
        self,
        chunks: list[RetrievalResult] | list[dict[str, Any]],
    ) -> str:
        """
        Format evidence chunks for prompt.

        Args:
            chunks: Evidence chunks to format

        Returns:
            Formatted evidence string
        """
        if not chunks:
            return "No evidence chunks available."
        
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            # Handle both RetrievalResult and dict
            if isinstance(chunk, dict):
                pmid = chunk.get("pmid", "unknown")
                chunk_id = chunk.get("chunk_id", f"{pmid}_{i}")
                text = chunk.get("text", "")
                score = chunk.get("score", chunk.get("rerank_score"))
            else:
                pmid = chunk.pmid
                chunk_id = chunk.chunk_id
                text = chunk.text
                score = chunk.rerank_score if chunk.rerank_score is not None else chunk.score
            
            # Format chunk with citation info
            score_str = f" (score: {score:.3f})" if score is not None else ""
            formatted_chunks.append(
                f"[{i}] PMID: {pmid} | Chunk: {chunk_id}{score_str}\n{text}"
            )
        
        return "\n\n".join(formatted_chunks)

    def get_prompt_hash(
        self,
        question: str,
        evidence_chunks: list[RetrievalResult] | list[dict[str, Any]],
        question_type: str = "factoid",
        **kwargs: Any,
    ) -> str:
        """
        Get stable hash of a rendered prompt.
        
        Used for caching LLM outputs.

        Args:
            question: The question
            evidence_chunks: Evidence chunks
            question_type: Question type
            **kwargs: Additional variables

        Returns:
            SHA256 hash of the prompt
        """
        rendered = self.render(question, evidence_chunks, question_type, **kwargs)
        return hashlib.sha256(rendered.encode()).hexdigest()


class PromptManager:
    """
    Manages multiple prompt templates.
    
    Provides caching and easy access to templates by name.
    """

    def __init__(self, configs_dir: Path | None = None) -> None:
        """
        Initialize prompt manager.

        Args:
            configs_dir: Base directory for configs
        """
        if configs_dir is None:
            configs_dir = Path(__file__).parent.parent.parent.parent / "configs"
        
        self.configs_dir = Path(configs_dir)
        self._templates: dict[str, PromptTemplate] = {}

    def get_template(self, template_path: str) -> PromptTemplate:
        """
        Get or create a template by path.

        Args:
            template_path: Path to template file

        Returns:
            PromptTemplate instance
        """
        if template_path not in self._templates:
            self._templates[template_path] = PromptTemplate(
                template_path, self.configs_dir
            )
        return self._templates[template_path]

    def list_templates(self) -> list[str]:
        """
        List available template files.

        Returns:
            List of template file paths
        """
        prompts_dir = self.configs_dir / "prompts"
        if not prompts_dir.exists():
            return []
        
        templates = []
        for path in prompts_dir.glob("*.txt"):
            templates.append(f"prompts/{path.name}")
        
        return sorted(templates)

    def render(
        self,
        template_path: str,
        question: str,
        evidence_chunks: list[RetrievalResult] | list[dict[str, Any]],
        question_type: str = "factoid",
        **kwargs: Any,
    ) -> str:
        """
        Render a template by path.

        Args:
            template_path: Path to template file
            question: The question
            evidence_chunks: Evidence chunks
            question_type: Question type
            **kwargs: Additional variables

        Returns:
            Rendered prompt
        """
        template = self.get_template(template_path)
        return template.render(question, evidence_chunks, question_type, **kwargs)





