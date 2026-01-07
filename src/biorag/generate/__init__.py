"""Generation module for BioRAG Bench."""

from biorag.generate.abstention import (
    AbstentionChecker,
    AbstentionConfig,
    AbstentionDecision,
    AbstentionReason,
    apply_abstention,
)
from biorag.generate.generator import (
    GenerationError,
    Generator,
)
from biorag.generate.prompts import (
    PromptManager,
    PromptTemplate,
)

__all__ = [
    # Abstention
    "AbstentionChecker",
    "AbstentionConfig",
    "AbstentionDecision",
    "AbstentionReason",
    "apply_abstention",
    # Generator
    "GenerationError",
    "Generator",
    # Prompts
    "PromptManager",
    "PromptTemplate",
]






