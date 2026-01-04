"""Generator module for BioRAG Bench with structured LLM outputs."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from biorag.generate.abstention import (
    AbstentionChecker,
    AbstentionConfig,
    AbstentionDecision,
    apply_abstention,
)
from biorag.generate.prompts import PromptManager, PromptTemplate
from biorag.schemas.config import BioRAGConfig, LLMConfig
from biorag.schemas.evaluation import RetrievalResult
from biorag.schemas.generation import (
    AnswerOutput,
    Citation,
    GenerationRequest,
    GenerationResponse,
)
from biorag.utils.caching import LLMCache
from biorag.utils.cost import CostTracker, count_tokens
from biorag.utils.logging import get_logger

logger = get_logger(__name__)


class GenerationError(Exception):
    """Raised when generation fails."""

    def __init__(
        self,
        message: str,
        retries_attempted: int = 0,
        last_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.retries_attempted = retries_attempted
        self.last_error = last_error


class Generator:
    """
    LLM generator with structured outputs, caching, and abstention.
    
    Features:
    - Structured JSON output with Pydantic validation
    - Retry logic on validation failure
    - LLM output caching
    - Abstention logic integration
    - Cost tracking
    """

    def __init__(
        self,
        config: BioRAGConfig | None = None,
        llm_config: LLMConfig | None = None,
        cache: LLMCache | None = None,
        cost_tracker: CostTracker | None = None,
        prompt_manager: PromptManager | None = None,
    ) -> None:
        """
        Initialize generator.

        Args:
            config: Full BioRAG config (takes precedence)
            llm_config: LLM-specific config
            cache: LLM output cache
            cost_tracker: Cost tracker instance
            prompt_manager: Prompt template manager
        """
        if config is not None:
            self.llm_config = config.llm
            self.prompt_config = config.prompt
            self.abstention_config = AbstentionConfig(
                min_evidence_score=config.abstention.min_evidence_score,
                min_evidence_chunks=config.abstention.min_evidence_chunks,
                enable_self_check=config.abstention.enable_self_check,
            )
            cache_dir = config.paths.cache_dir
        else:
            self.llm_config = llm_config or LLMConfig()
            self.prompt_config = None
            self.abstention_config = AbstentionConfig()
            cache_dir = Path("data/cache")
        
        # Initialize components
        self.cache = cache or LLMCache(cache_dir=cache_dir)
        self.cost_tracker = cost_tracker
        self.prompt_manager = prompt_manager or PromptManager()
        self.abstention_checker = AbstentionChecker(self.abstention_config)
        
        # LLM client (lazy initialization)
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Get or create OpenAI client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> Any:
        """Create OpenAI client."""
        from openai import OpenAI
        
        return OpenAI()

    def generate(
        self,
        request: GenerationRequest,
        template_path: str | None = None,
        use_cache: bool = True,
    ) -> GenerationResponse:
        """
        Generate an answer for a question.

        Args:
            request: Generation request with question and evidence
            template_path: Path to prompt template
            use_cache: Whether to use caching

        Returns:
            GenerationResponse with answer and metadata
        """
        start_time = time.perf_counter()
        
        # Get template path
        if template_path is None:
            template_path = (
                self.prompt_config.template
                if self.prompt_config
                else "prompts/cite_and_abstain_v2.txt"
            )
        
        # Convert evidence to proper format
        evidence_chunks = self._normalize_evidence(request.evidence_chunks)
        
        # Pre-generation abstention check
        abstention_decision = self.abstention_checker.check_evidence(evidence_chunks)
        if abstention_decision.should_abstain:
            return self._create_abstention_response(
                abstention_decision,
                template_path,
                start_time,
            )
        
        # Render prompt
        template = self.prompt_manager.get_template(template_path)
        prompt = template.render(
            question=request.question,
            evidence_chunks=evidence_chunks,
            question_type=request.question_type or "factoid",
        )
        
        # Check cache
        cache_key = LLMCache.compute_cache_key(
            model=self.llm_config.model,
            prompt=prompt,
            template_hash=template.template_hash,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
        )
        
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug("Cache hit for generation")
                output = AnswerOutput.model_validate(cached["response"])
                
                # Track cost (as cache hit)
                if self.cost_tracker:
                    self.cost_tracker.add_request(
                        cached["input_tokens"],
                        cached["output_tokens"],
                        is_cache_hit=True,
                    )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                return GenerationResponse(
                    answer=output,
                    model=self.llm_config.model,
                    prompt_template=template_path,
                    input_tokens=cached["input_tokens"],
                    output_tokens=cached["output_tokens"],
                    latency_ms=latency_ms,
                    cache_hit=True,
                )
        
        # Call LLM with retries
        output, input_tokens, output_tokens = self._call_llm_with_retry(
            prompt=prompt,
            question_type=request.question_type,
        )
        
        # Post-generation abstention check
        final_decision = self.abstention_checker.check_model_output(
            output, abstention_decision
        )
        if final_decision.should_abstain:
            output = apply_abstention(output, final_decision)
        
        # Validate citations
        if request.require_citations and not output.abstained:
            self._validate_citations(output, evidence_chunks)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Cache the response
        if use_cache and not output.abstained:
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            self.cache.set(
                cache_key=cache_key,
                response=output.model_dump(),
                model=self.llm_config.model,
                prompt_hash=prompt_hash,
                template_hash=template.template_hash,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )
        
        # Track cost
        if self.cost_tracker:
            self.cost_tracker.add_request(
                input_tokens,
                output_tokens,
                is_cache_hit=False,
            )
        
        # Log generation
        logger.log_generation(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model=self.llm_config.model,
            abstained=output.abstained,
        )
        
        return GenerationResponse(
            answer=output,
            model=self.llm_config.model,
            prompt_template=template_path,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cache_hit=False,
        )

    def _call_llm_with_retry(
        self,
        prompt: str,
        question_type: str | None = None,
    ) -> tuple[AnswerOutput, int, int]:
        """
        Call LLM with retry logic on validation failure.

        Args:
            prompt: Rendered prompt
            question_type: Type of question for validation

        Returns:
            Tuple of (AnswerOutput, input_tokens, output_tokens)

        Raises:
            GenerationError: If all retries fail
        """
        last_error: Exception | None = None
        
        for attempt in range(self.llm_config.max_retries + 1):
            try:
                return self._call_llm(prompt, question_type)
            except (ValidationError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning(
                    f"Generation attempt {attempt + 1} failed: {e}"
                )
                
                # On retry, we could modify the prompt, but for now just retry
                continue
            except Exception as e:
                # Re-raise unexpected errors
                raise GenerationError(
                    f"LLM call failed: {e}",
                    retries_attempted=attempt,
                    last_error=e,
                ) from e
        
        # All retries exhausted - return unknown answer
        logger.error(f"All {self.llm_config.max_retries + 1} attempts failed")
        
        # Estimate tokens for the failed request
        input_tokens = count_tokens(prompt, self.llm_config.model)
        
        return (
            AnswerOutput(
                answer="Unable to generate a valid response.",
                answer_type="unknown",
                abstained=True,
                abstention_reason=f"Generation failed after {self.llm_config.max_retries + 1} attempts: {last_error}",
                supported_by_evidence=False,
            ),
            input_tokens,
            0,
        )

    def _call_llm(
        self,
        prompt: str,
        question_type: str | None = None,
    ) -> tuple[AnswerOutput, int, int]:
        """
        Make a single LLM call.

        Args:
            prompt: Rendered prompt
            question_type: Type of question

        Returns:
            Tuple of (AnswerOutput, input_tokens, output_tokens)
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Use JSON mode for structured output
        response = self.client.chat.completions.create(
            model=self.llm_config.model,
            messages=messages,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
            timeout=self.llm_config.timeout,
            response_format={"type": "json_object"},
        )
        
        # Extract response
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from LLM")
        
        # Parse JSON from response
        parsed = self._parse_json_response(content)
        
        # Validate and create output
        output = self._create_answer_output(parsed, question_type)
        
        # Get token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        
        return output, input_tokens, output_tokens

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        """
        Parse JSON from LLM response.
        
        Handles cases where JSON is wrapped in markdown code blocks.

        Args:
            content: Raw response content

        Returns:
            Parsed JSON dict
        """
        # Try direct parsing first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Try finding JSON object
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return json.loads(json_match.group(0))
        
        raise json.JSONDecodeError("No valid JSON found", content, 0)

    def _create_answer_output(
        self,
        parsed: dict[str, Any],
        question_type: str | None = None,
    ) -> AnswerOutput:
        """
        Create AnswerOutput from parsed JSON.

        Args:
            parsed: Parsed JSON dict
            question_type: Question type for validation

        Returns:
            Validated AnswerOutput
        """
        # Extract citations
        citations = []
        for cit in parsed.get("citations", []):
            if isinstance(cit, dict):
                citations.append(Citation(
                    pmid=str(cit.get("pmid", "")),
                    chunk_id=cit.get("chunk_id"),
                    quote=cit.get("quote"),
                ))
        
        # Determine answer type
        if parsed.get("abstained", False):
            answer_type = "abstained"
        elif question_type == "summary":
            answer_type = "direct"
        else:
            answer_type = "direct"
        
        # Handle "null" string returned by LLM (should be None)
        label = parsed.get("label")
        if label == "null" or label == "":
            label = None
        
        abstention_reason = parsed.get("abstention_reason")
        if abstention_reason == "null" or abstention_reason == "":
            abstention_reason = None
        
        answer_list = parsed.get("answer_list")
        if answer_list == "null":
            answer_list = None
        
        return AnswerOutput(
            answer=parsed.get("answer", ""),
            answer_type=answer_type,
            label=label,
            confidence=parsed.get("confidence"),
            citations=citations,
            abstained=parsed.get("abstained", False),
            abstention_reason=abstention_reason,
            supported_by_evidence=parsed.get("supported_by_evidence", True),
            answer_list=answer_list,
        )

    def _validate_citations(
        self,
        output: AnswerOutput,
        evidence_chunks: list[RetrievalResult],
    ) -> None:
        """
        Validate that citations reference actual evidence.

        Args:
            output: Generated output
            evidence_chunks: Evidence chunks used

        Logs warning if citations are invalid.
        """
        valid_pmids = {chunk.pmid for chunk in evidence_chunks}
        valid_chunk_ids = {chunk.chunk_id for chunk in evidence_chunks}
        
        for citation in output.citations:
            if citation.pmid not in valid_pmids:
                logger.warning(f"Citation PMID {citation.pmid} not in evidence")
            if citation.chunk_id and citation.chunk_id not in valid_chunk_ids:
                logger.warning(f"Citation chunk_id {citation.chunk_id} not in evidence")

    def _normalize_evidence(
        self,
        evidence: list[dict[str, Any]] | list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Convert evidence to RetrievalResult format."""
        if not evidence:
            return []
        
        if isinstance(evidence[0], RetrievalResult):
            return evidence  # type: ignore
        
        # Convert dicts to RetrievalResult
        results = []
        for i, chunk in enumerate(evidence):
            results.append(RetrievalResult(
                pmid=chunk.get("pmid", "unknown"),
                chunk_id=chunk.get("chunk_id", f"chunk_{i}"),
                text=chunk.get("text", ""),
                score=chunk.get("score", 0.0),
                rank=chunk.get("rank", i + 1),
                rerank_score=chunk.get("rerank_score"),
                rerank_rank=chunk.get("rerank_rank"),
            ))
        
        return results

    def _create_abstention_response(
        self,
        decision: AbstentionDecision,
        template_path: str,
        start_time: float,
    ) -> GenerationResponse:
        """Create a response for pre-generation abstention."""
        output = AnswerOutput(
            answer="I cannot answer this question based on the available evidence.",
            answer_type="abstained",
            label=None,
            confidence=0.0,
            citations=[],
            abstained=True,
            abstention_reason=decision.details or (decision.reason.value if decision.reason else "Unknown"),
            supported_by_evidence=False,
            answer_list=None,
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return GenerationResponse(
            answer=output,
            model=self.llm_config.model,
            prompt_template=template_path,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            cache_hit=False,
        )

    def generate_batch(
        self,
        requests: list[GenerationRequest],
        template_path: str | None = None,
        use_cache: bool = True,
    ) -> list[GenerationResponse]:
        """
        Generate answers for multiple questions.
        
        Note: Currently sequential, could be parallelized in future.

        Args:
            requests: List of generation requests
            template_path: Path to prompt template
            use_cache: Whether to use caching

        Returns:
            List of GenerationResponse objects
        """
        responses = []
        for request in requests:
            # Check if we should skip due to budget
            if self.cost_tracker and self.cost_tracker.should_skip():
                responses.append(self._create_budget_skip_response(template_path or ""))
                continue
            
            response = self.generate(request, template_path, use_cache)
            responses.append(response)
        
        return responses

    def _create_budget_skip_response(self, template_path: str) -> GenerationResponse:
        """Create response for budget-skipped question."""
        return GenerationResponse(
            answer=AnswerOutput(
                answer="Skipped due to budget constraints.",
                answer_type="abstained",
                abstained=True,
                abstention_reason="Budget limit reached",
                supported_by_evidence=False,
            ),
            model=self.llm_config.model,
            prompt_template=template_path,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            cache_hit=False,
        )

