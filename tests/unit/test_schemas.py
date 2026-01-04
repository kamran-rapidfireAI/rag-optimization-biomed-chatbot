"""Unit tests for schema modules."""

from __future__ import annotations

import pytest

from biorag.schemas.corpus import Chunk, CorpusDocument, CorpusManifest
from biorag.schemas.evaluation import (
    BioASQQuestion,
    EvalPrediction,
    MetricResult,
    PubMedQAQuestion,
    RetrievalResult,
    RunMetrics,
)
from biorag.schemas.generation import AnswerOutput, Citation, GenerationResponse


class TestCorpusDocument:
    """Tests for CorpusDocument schema."""

    def test_basic_document(self) -> None:
        """Test creating a basic corpus document."""
        doc = CorpusDocument(
            pmid="12345678",
            title="Test Article",
            abstract="This is a test abstract with biomedical content.",
        )
        assert doc.pmid == "12345678"
        assert doc.title == "Test Article"
        assert doc.is_gold is False
        assert doc.source == "pubmed"

    def test_full_text(self) -> None:
        """Test full_text method."""
        doc = CorpusDocument(
            pmid="12345678",
            title="Test Title",
            abstract="Test abstract content.",
        )
        assert "Test Title" in doc.full_text()
        assert "Test abstract content" in doc.full_text()

    def test_document_without_title(self) -> None:
        """Test document without title."""
        doc = CorpusDocument(pmid="12345678", abstract="Abstract only.")
        assert doc.full_text() == "Abstract only."

    def test_gold_document(self) -> None:
        """Test gold document flag."""
        doc = CorpusDocument(
            pmid="12345678",
            abstract="Gold abstract.",
            is_gold=True,
        )
        assert doc.is_gold is True


class TestChunk:
    """Tests for Chunk schema."""

    def test_basic_chunk(self) -> None:
        """Test creating a basic chunk."""
        chunk = Chunk(
            chunk_id="12345678_0",
            pmid="12345678",
            text="This is chunk text.",
        )
        assert chunk.chunk_id == "12345678_0"
        assert chunk.pmid == "12345678"
        assert chunk.doc_id == "12345678"

    def test_chunk_with_position(self) -> None:
        """Test chunk with position info."""
        chunk = Chunk(
            chunk_id="12345678_1",
            pmid="12345678",
            text="Second chunk.",
            start_char=100,
            end_char=200,
            chunk_index=1,
            total_chunks=3,
        )
        assert chunk.chunk_index == 1
        assert chunk.total_chunks == 3


class TestCorpusManifest:
    """Tests for CorpusManifest schema."""

    def test_default_manifest(self) -> None:
        """Test default manifest values."""
        manifest = CorpusManifest()
        assert manifest.source_method == "huggingface"
        assert manifest.sampling_seed == 42

    def test_manifest_with_counts(self) -> None:
        """Test manifest with counts."""
        manifest = CorpusManifest(
            gold_pmid_count=100,
            distractor_pmid_count=10000,
            total_records=10100,
        )
        assert manifest.gold_pmid_count == 100
        assert manifest.total_records == 10100


class TestBioASQQuestion:
    """Tests for BioASQQuestion schema."""

    def test_yesno_question(self) -> None:
        """Test yes/no question."""
        q = BioASQQuestion(
            question_id="bioasq_1",
            question_text="Is aspirin effective for pain?",
            question_type="yesno",
            exact_answer="yes",
        )
        assert q.question_type == "yesno"
        assert q.exact_answer == "yes"

    def test_factoid_question(self) -> None:
        """Test factoid question."""
        q = BioASQQuestion(
            question_id="bioasq_2",
            question_text="What is the target of imatinib?",
            question_type="factoid",
            exact_answer="BCR-ABL",
            gold_pmids=["12345678", "87654321"],
        )
        assert q.question_type == "factoid"
        assert len(q.gold_pmids) == 2

    def test_list_question(self) -> None:
        """Test list question."""
        q = BioASQQuestion(
            question_id="bioasq_3",
            question_text="What are the symptoms of COVID-19?",
            question_type="list",
            exact_answer=["fever", "cough", "fatigue"],
        )
        assert q.question_type == "list"
        assert isinstance(q.exact_answer, list)
        assert len(q.exact_answer) == 3

    def test_summary_question(self) -> None:
        """Test summary question."""
        q = BioASQQuestion(
            question_id="bioasq_4",
            question_text="Describe the mechanism of action of metformin.",
            question_type="summary",
            ideal_answer="Metformin works by...",
        )
        assert q.question_type == "summary"
        assert q.ideal_answer is not None


class TestPubMedQAQuestion:
    """Tests for PubMedQAQuestion schema."""

    def test_basic_question(self) -> None:
        """Test basic PubMedQA question."""
        q = PubMedQAQuestion(
            question_id="pubmedqa_12345",
            question_text="Does drug X improve outcomes?",
            label="yes",
            pmid="12345",
        )
        assert q.label == "yes"
        assert q.pmid == "12345"

    def test_question_with_context(self) -> None:
        """Test question with context."""
        q = PubMedQAQuestion(
            question_id="pubmedqa_12345",
            question_text="Is treatment effective?",
            context=["Background sentence.", "Methods used.", "Results found."],
            long_answer="The treatment was found to be effective.",
            label="yes",
            pmid="12345",
        )
        assert len(q.context) == 3
        assert q.long_answer != ""


class TestAnswerOutput:
    """Tests for AnswerOutput schema."""

    def test_direct_answer(self) -> None:
        """Test direct answer."""
        answer = AnswerOutput(
            answer="The answer is 42.",
            answer_type="direct",
        )
        assert answer.answer_type == "direct"
        assert answer.abstained is False

    def test_abstained_answer(self) -> None:
        """Test abstained answer."""
        answer = AnswerOutput(
            answer="I cannot answer this question.",
            answer_type="abstained",
            abstained=True,
            abstention_reason="Insufficient evidence",
        )
        assert answer.abstained is True
        assert answer.abstention_reason is not None

    def test_answer_with_citations(self) -> None:
        """Test answer with citations."""
        answer = AnswerOutput(
            answer="Drug X is effective [1].",
            citations=[
                Citation(pmid="12345678", chunk_id="12345678_0"),
                Citation(pmid="87654321", chunk_id="87654321_1"),
            ],
        )
        assert len(answer.citations) == 2

    def test_has_valid_citations(self) -> None:
        """Test citation validation."""
        answer = AnswerOutput(
            answer="Test answer.",
            citations=[Citation(pmid="12345678")],
        )
        assert answer.has_valid_citations()
        assert answer.has_valid_citations({"12345678"})
        assert not answer.has_valid_citations({"99999999"})

    def test_list_answer(self) -> None:
        """Test list-type answer."""
        answer = AnswerOutput(
            answer="The symptoms include: fever, cough, fatigue.",
            answer_list=["fever", "cough", "fatigue"],
        )
        assert answer.answer_list is not None
        assert len(answer.answer_list) == 3


class TestRetrievalResult:
    """Tests for RetrievalResult schema."""

    def test_basic_result(self) -> None:
        """Test basic retrieval result."""
        result = RetrievalResult(
            pmid="12345678",
            chunk_id="12345678_0",
            text="Retrieved text content.",
            score=0.85,
            rank=1,
        )
        assert result.score == 0.85
        assert result.rank == 1

    def test_reranked_result(self) -> None:
        """Test result with reranking scores."""
        result = RetrievalResult(
            pmid="12345678",
            chunk_id="12345678_0",
            text="Retrieved text.",
            score=0.75,
            rank=3,
            rerank_score=0.92,
            rerank_rank=1,
        )
        assert result.rerank_score == 0.92
        assert result.rerank_rank == 1


class TestEvalPrediction:
    """Tests for EvalPrediction schema."""

    def test_basic_prediction(self) -> None:
        """Test basic prediction."""
        pred = EvalPrediction(
            question_id="q1",
            predicted_answer="The answer is X.",
            total_latency_ms=150.0,
        )
        assert pred.question_id == "q1"
        assert pred.abstained is False

    def test_prediction_with_retrieval(self) -> None:
        """Test prediction with retrieval info."""
        pred = EvalPrediction(
            question_id="q1",
            retrieved_pmids=["12345", "67890"],
            predicted_answer="Based on the evidence...",
            retrieval_latency_ms=50.0,
            generation_latency_ms=100.0,
        )
        assert len(pred.retrieved_pmids) == 2


class TestRunMetrics:
    """Tests for RunMetrics schema."""

    def test_basic_metrics(self) -> None:
        """Test basic run metrics."""
        metrics = RunMetrics(
            run_id="run_001",
            dataset="bioasq",
            num_questions=100,
        )
        assert metrics.run_id == "run_001"
        assert metrics.num_questions == 100

    def test_metrics_with_results(self) -> None:
        """Test metrics with actual results."""
        metrics = RunMetrics(
            run_id="run_001",
            dataset="bioasq",
            num_questions=100,
            retrieval_metrics={
                "recall_at_10": MetricResult(name="recall_at_10", value=0.85, count=100),
                "mrr": MetricResult(name="mrr", value=0.72, count=100),
            },
            answer_metrics={
                "exact_match": MetricResult(name="exact_match", value=0.45, count=100),
                "token_f1": MetricResult(name="token_f1", value=0.68, count=100),
            },
        )
        assert metrics.retrieval_metrics["recall_at_10"].value == 0.85
        assert metrics.answer_metrics["token_f1"].value == 0.68

