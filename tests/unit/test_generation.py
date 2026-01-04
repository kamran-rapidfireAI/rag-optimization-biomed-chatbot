"""Unit tests for generation module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from biorag.generate.abstention import (
    AbstentionChecker,
    AbstentionConfig,
    AbstentionDecision,
    AbstentionReason,
    apply_abstention,
)
from biorag.generate.generator import Generator, GenerationError
from biorag.generate.prompts import PromptManager, PromptTemplate
from biorag.schemas.config import LLMConfig
from biorag.schemas.evaluation import RetrievalResult
from biorag.schemas.generation import AnswerOutput, GenerationRequest


# ============================================================================
# Prompt Template Tests
# ============================================================================


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    @pytest.fixture
    def configs_dir(self, tmp_path: Path) -> Path:
        """Create temporary configs directory with test templates."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        
        # Create a simple test template
        template = prompts_dir / "test_template.txt"
        template.write_text(
            "Question: {question}\n"
            "Evidence:\n{evidence}\n"
            "Type: {question_type}"
        )
        
        return tmp_path

    def test_load_template(self, configs_dir: Path) -> None:
        """Test loading a template from file."""
        template = PromptTemplate(
            "prompts/test_template.txt",
            configs_dir=configs_dir,
        )
        
        assert "Question: {question}" in template.template
        assert template.template_hash is not None
        assert len(template.template_hash) == 16

    def test_template_not_found(self, tmp_path: Path) -> None:
        """Test error when template not found."""
        template = PromptTemplate(
            "nonexistent.txt",
            configs_dir=tmp_path,
        )
        
        with pytest.raises(FileNotFoundError):
            _ = template.template

    def test_render_basic(self, configs_dir: Path) -> None:
        """Test basic template rendering."""
        template = PromptTemplate(
            "prompts/test_template.txt",
            configs_dir=configs_dir,
        )
        
        chunks = [
            RetrievalResult(
                pmid="12345678",
                chunk_id="12345678_0",
                text="Test evidence text",
                score=0.95,
                rank=1,
            )
        ]
        
        rendered = template.render(
            question="What is the treatment?",
            evidence_chunks=chunks,
            question_type="factoid",
        )
        
        assert "What is the treatment?" in rendered
        assert "Test evidence text" in rendered
        assert "12345678" in rendered
        assert "factoid" in rendered

    def test_render_with_dicts(self, configs_dir: Path) -> None:
        """Test rendering with dict evidence chunks."""
        template = PromptTemplate(
            "prompts/test_template.txt",
            configs_dir=configs_dir,
        )
        
        chunks = [
            {
                "pmid": "87654321",
                "chunk_id": "87654321_0",
                "text": "Dict evidence",
                "score": 0.8,
            }
        ]
        
        rendered = template.render(
            question="Test question",
            evidence_chunks=chunks,
            question_type="yesno",
        )
        
        assert "Dict evidence" in rendered
        assert "87654321" in rendered

    def test_render_empty_evidence(self, configs_dir: Path) -> None:
        """Test rendering with no evidence."""
        template = PromptTemplate(
            "prompts/test_template.txt",
            configs_dir=configs_dir,
        )
        
        rendered = template.render(
            question="Test question",
            evidence_chunks=[],
            question_type="factoid",
        )
        
        assert "No evidence chunks available" in rendered

    def test_render_template_with_json_braces(self, tmp_path: Path) -> None:
        """Test that templates with escaped JSON braces render correctly.
        
        This tests the fix for the bug where literal {} in JSON examples
        were interpreted as template variables by Python's str.format().
        Templates must use {{ and }} to escape literal braces.
        """
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        
        # Create template with escaped JSON braces (like the real templates)
        template_content = '''Question: {question}
Evidence: {evidence}

Respond with JSON:
```json
{{
    "answer": "Your answer here",
    "citations": [{{"pmid": "12345"}}],
    "abstained": false
}}
```
'''
        template_file = prompts_dir / "json_template.txt"
        template_file.write_text(template_content)
        
        template = PromptTemplate(
            "prompts/json_template.txt",
            configs_dir=tmp_path,
        )
        
        chunks = [
            RetrievalResult(
                pmid="99999",
                chunk_id="99999_0",
                text="Test evidence",
                score=0.9,
                rank=1,
            )
        ]
        
        # This should not raise KeyError
        rendered = template.render(
            question="What is X?",
            evidence_chunks=chunks,
            question_type="factoid",
        )
        
        # Verify template variables were substituted
        assert "What is X?" in rendered
        assert "Test evidence" in rendered
        
        # Verify JSON braces are properly unescaped in output
        assert '"answer": "Your answer here"' in rendered
        assert '"pmid": "12345"' in rendered
        assert "{{" not in rendered  # Escaped braces should be unescaped
        assert "}}" not in rendered

    def test_real_templates_render_correctly(self) -> None:
        """Test that the actual prompt templates can be rendered.
        
        This is a regression test for the escaped JSON braces fix.
        """
        pm = PromptManager()
        
        # Test both real templates
        for template_path in ["prompts/cite_and_abstain_v1.txt", "prompts/cite_and_abstain_v2.txt"]:
            template = pm.get_template(template_path)
            
            chunks = [
                RetrievalResult(
                    pmid="12345678",
                    chunk_id="12345678_0",
                    text="BRCA1 is a tumor suppressor gene.",
                    score=0.95,
                    rank=1,
                )
            ]
            
            # Should not raise any errors
            rendered = template.render(
                question="What is BRCA1?",
                evidence_chunks=chunks,
                question_type="factoid",
            )
            
            # Verify key parts are present
            assert "What is BRCA1?" in rendered
            assert "BRCA1 is a tumor suppressor gene." in rendered
            assert "12345678" in rendered
            # Verify JSON example is properly rendered (braces unescaped)
            assert '"answer"' in rendered
            assert '"citations"' in rendered

    def test_get_prompt_hash(self, configs_dir: Path) -> None:
        """Test that same inputs produce same hash."""
        template = PromptTemplate(
            "prompts/test_template.txt",
            configs_dir=configs_dir,
        )
        
        chunks = [
            RetrievalResult(
                pmid="12345678",
                chunk_id="12345678_0",
                text="Evidence",
                score=0.9,
                rank=1,
            )
        ]
        
        hash1 = template.get_prompt_hash("Question", chunks, "factoid")
        hash2 = template.get_prompt_hash("Question", chunks, "factoid")
        hash3 = template.get_prompt_hash("Different", chunks, "factoid")
        
        assert hash1 == hash2
        assert hash1 != hash3


class TestPromptManager:
    """Tests for PromptManager class."""

    @pytest.fixture
    def configs_dir(self, tmp_path: Path) -> Path:
        """Create temporary configs with multiple templates."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        
        (prompts_dir / "template1.txt").write_text("Template 1: {question}")
        (prompts_dir / "template2.txt").write_text("Template 2: {question}")
        
        return tmp_path

    def test_get_template(self, configs_dir: Path) -> None:
        """Test getting a template."""
        manager = PromptManager(configs_dir=configs_dir)
        
        template = manager.get_template("prompts/template1.txt")
        assert template is not None
        
        # Same template should be cached
        template2 = manager.get_template("prompts/template1.txt")
        assert template is template2

    def test_list_templates(self, configs_dir: Path) -> None:
        """Test listing available templates."""
        manager = PromptManager(configs_dir=configs_dir)
        
        templates = manager.list_templates()
        
        assert "prompts/template1.txt" in templates
        assert "prompts/template2.txt" in templates

    def test_render(self, configs_dir: Path) -> None:
        """Test rendering through manager."""
        manager = PromptManager(configs_dir=configs_dir)
        
        rendered = manager.render(
            "prompts/template1.txt",
            question="Test",
            evidence_chunks=[],
            question_type="factoid",
        )
        
        assert "Template 1: Test" in rendered


# ============================================================================
# Abstention Tests
# ============================================================================


class TestAbstentionChecker:
    """Tests for AbstentionChecker class."""

    def test_check_no_evidence(self) -> None:
        """Test abstention when no evidence."""
        checker = AbstentionChecker()
        
        decision = checker.check_evidence([])
        
        assert decision.should_abstain
        assert decision.reason == AbstentionReason.NO_EVIDENCE

    def test_check_insufficient_chunks(self) -> None:
        """Test abstention when not enough chunks."""
        config = AbstentionConfig(min_evidence_chunks=3)
        checker = AbstentionChecker(config)
        
        chunks = [
            RetrievalResult(pmid="1", chunk_id="1_0", text="t", score=0.9, rank=1),
            RetrievalResult(pmid="2", chunk_id="2_0", text="t", score=0.8, rank=2),
        ]
        
        decision = checker.check_evidence(chunks)
        
        assert decision.should_abstain
        assert decision.reason == AbstentionReason.INSUFFICIENT_CHUNKS

    def test_check_low_score(self) -> None:
        """Test abstention when scores too low."""
        config = AbstentionConfig(min_evidence_score=0.5)
        checker = AbstentionChecker(config)
        
        chunks = [
            RetrievalResult(pmid="1", chunk_id="1_0", text="t", score=0.3, rank=1),
        ]
        
        decision = checker.check_evidence(chunks)
        
        assert decision.should_abstain
        assert decision.reason == AbstentionReason.LOW_EVIDENCE_SCORE

    def test_check_evidence_passes(self) -> None:
        """Test evidence passes checks."""
        config = AbstentionConfig(min_evidence_score=0.3, min_evidence_chunks=1)
        checker = AbstentionChecker(config)
        
        chunks = [
            RetrievalResult(pmid="1", chunk_id="1_0", text="t", score=0.9, rank=1),
        ]
        
        decision = checker.check_evidence(chunks)
        
        assert not decision.should_abstain

    def test_check_model_output_abstained(self) -> None:
        """Test respecting model's abstention."""
        checker = AbstentionChecker()
        
        output = AnswerOutput(
            answer="Cannot answer",
            abstained=True,
            abstention_reason="Evidence unclear",
        )
        
        decision = checker.check_model_output(output)
        
        assert decision.should_abstain
        assert decision.reason == AbstentionReason.MODEL_UNCERTAIN

    def test_check_model_output_unsupported(self) -> None:
        """Test abstention when model reports unsupported."""
        config = AbstentionConfig(enable_self_check=True)
        checker = AbstentionChecker(config)
        
        output = AnswerOutput(
            answer="Some answer",
            abstained=False,
            supported_by_evidence=False,
        )
        
        decision = checker.check_model_output(output)
        
        assert decision.should_abstain
        assert decision.reason == AbstentionReason.UNSUPPORTED_BY_EVIDENCE

    def test_check_model_output_valid(self) -> None:
        """Test valid output passes."""
        checker = AbstentionChecker()
        
        output = AnswerOutput(
            answer="Valid answer",
            abstained=False,
            supported_by_evidence=True,
        )
        
        decision = checker.check_model_output(output)
        
        assert not decision.should_abstain


class TestApplyAbstention:
    """Tests for apply_abstention function."""

    def test_apply_no_abstention(self) -> None:
        """Test output unchanged when not abstaining."""
        output = AnswerOutput(answer="Original")
        decision = AbstentionDecision(should_abstain=False)
        
        result = apply_abstention(output, decision)
        
        assert result.answer == "Original"
        assert not result.abstained

    def test_apply_abstention(self) -> None:
        """Test abstention is applied."""
        output = AnswerOutput(answer="Original")
        decision = AbstentionDecision(
            should_abstain=True,
            reason=AbstentionReason.LOW_EVIDENCE_SCORE,
            details="Score too low",
        )
        
        result = apply_abstention(output, decision)
        
        assert result.abstained
        assert result.answer_type == "abstained"
        assert "cannot answer" in result.answer.lower()
        assert result.abstention_reason == "Score too low"


# ============================================================================
# Generator Tests
# ============================================================================


class TestGenerator:
    """Tests for Generator class."""

    @pytest.fixture
    def mock_openai(self) -> MagicMock:
        """Create mock OpenAI client."""
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps({
                        "answer": "The treatment is X",
                        "label": None,
                        "answer_list": None,
                        "citations": [
                            {"pmid": "12345678", "chunk_id": "12345678_0", "quote": "X is effective"}
                        ],
                        "supported_by_evidence": True,
                        "abstained": False,
                        "abstention_reason": None,
                    })
                )
            )
        ]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        return mock_client

    @pytest.fixture
    def configs_dir(self, tmp_path: Path) -> Path:
        """Create temporary configs."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        
        template = prompts_dir / "test.txt"
        template.write_text(
            "Question: {question}\nEvidence: {evidence}\nType: {question_type}"
        )
        
        return tmp_path

    def test_generate_success(
        self,
        mock_openai: MagicMock,
        configs_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test successful generation."""
        from biorag.generate.prompts import PromptManager
        from biorag.utils.caching import LLMCache
        
        cache = LLMCache(cache_dir=tmp_path / "cache")
        prompt_manager = PromptManager(configs_dir=configs_dir)
        
        generator = Generator(
            llm_config=LLMConfig(),
            cache=cache,
            prompt_manager=prompt_manager,
        )
        generator._client = mock_openai
        
        request = GenerationRequest(
            question="What is the treatment?",
            evidence_chunks=[
                {"pmid": "12345678", "chunk_id": "12345678_0", "text": "Evidence", "score": 0.9}
            ],
            question_type="factoid",
        )
        
        response = generator.generate(
            request,
            template_path="prompts/test.txt",
            use_cache=False,
        )
        
        assert response.answer.answer == "The treatment is X"
        assert not response.answer.abstained
        assert len(response.answer.citations) == 1
        assert response.input_tokens == 100
        assert response.output_tokens == 50

    def test_generate_abstains_no_evidence(
        self,
        configs_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test abstention when no evidence."""
        from biorag.generate.prompts import PromptManager
        from biorag.utils.caching import LLMCache
        
        cache = LLMCache(cache_dir=tmp_path / "cache")
        prompt_manager = PromptManager(configs_dir=configs_dir)
        
        generator = Generator(
            llm_config=LLMConfig(),
            cache=cache,
            prompt_manager=prompt_manager,
        )
        
        request = GenerationRequest(
            question="What is the treatment?",
            evidence_chunks=[],
            question_type="factoid",
        )
        
        response = generator.generate(
            request,
            template_path="prompts/test.txt",
        )
        
        assert response.answer.abstained
        assert response.answer.answer_type == "abstained"
        assert response.input_tokens == 0  # No LLM call made

    def test_parse_json_response_direct(self) -> None:
        """Test parsing direct JSON."""
        generator = Generator()
        
        content = '{"answer": "test", "citations": []}'
        result = generator._parse_json_response(content)
        
        assert result["answer"] == "test"

    def test_parse_json_response_code_block(self) -> None:
        """Test parsing JSON from code block."""
        generator = Generator()
        
        content = '```json\n{"answer": "test", "citations": []}\n```'
        result = generator._parse_json_response(content)
        
        assert result["answer"] == "test"

    def test_parse_json_response_embedded(self) -> None:
        """Test parsing embedded JSON."""
        generator = Generator()
        
        content = 'Here is my answer: {"answer": "test", "citations": []}'
        result = generator._parse_json_response(content)
        
        assert result["answer"] == "test"

    def test_create_answer_output(self) -> None:
        """Test creating AnswerOutput from parsed JSON."""
        generator = Generator()
        
        parsed = {
            "answer": "The answer is X",
            "label": "yes",
            "citations": [
                {"pmid": "123", "chunk_id": "123_0", "quote": "X is true"}
            ],
            "supported_by_evidence": True,
            "abstained": False,
        }
        
        output = generator._create_answer_output(parsed, "yesno")
        
        assert output.answer == "The answer is X"
        assert output.label == "yes"
        assert len(output.citations) == 1
        assert output.citations[0].pmid == "123"

    def test_normalize_evidence_retrieval_results(self) -> None:
        """Test normalizing RetrievalResult list."""
        generator = Generator()
        
        evidence = [
            RetrievalResult(pmid="1", chunk_id="1_0", text="t", score=0.9, rank=1),
        ]
        
        result = generator._normalize_evidence(evidence)
        
        assert result is evidence

    def test_normalize_evidence_dicts(self) -> None:
        """Test normalizing dict list."""
        generator = Generator()
        
        evidence = [
            {"pmid": "1", "chunk_id": "1_0", "text": "t", "score": 0.9},
        ]
        
        result = generator._normalize_evidence(evidence)
        
        assert len(result) == 1
        assert isinstance(result[0], RetrievalResult)
        assert result[0].pmid == "1"

