"""Unit tests for data loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biorag.data.bioasq_loader import BioASQLoader
from biorag.data.pubmedqa_loader import PubMedQALoader
from biorag.schemas.evaluation import BioASQQuestion, PubMedQAQuestion


class TestBioASQLoader:
    """Tests for BioASQLoader."""

    @pytest.fixture
    def sample_bioasq_data(self, temp_dir: Path) -> Path:
        """Create sample BioASQ data file."""
        data = {
            "questions": [
                {
                    "id": "bioasq_test_1",
                    "body": "Is aspirin effective for pain relief?",
                    "type": "yesno",
                    "documents": [
                        "http://www.ncbi.nlm.nih.gov/pubmed/12345678",
                        "http://www.ncbi.nlm.nih.gov/pubmed/87654321",
                    ],
                    "snippets": [
                        {"text": "Aspirin is widely used for pain relief."},
                        {"text": "Studies show aspirin effectiveness."},
                    ],
                    "exact_answer": "yes",
                },
                {
                    "id": "bioasq_test_2",
                    "body": "What is the target of imatinib?",
                    "type": "factoid",
                    "documents": ["http://www.ncbi.nlm.nih.gov/pubmed/11111111"],
                    "snippets": [{"text": "Imatinib targets BCR-ABL tyrosine kinase."}],
                    "exact_answer": [["BCR-ABL"]],
                },
                {
                    "id": "bioasq_test_3",
                    "body": "List common symptoms of influenza.",
                    "type": "list",
                    "documents": [],
                    "snippets": [],
                    "exact_answer": [["fever", "cough", "fatigue", "headache"]],
                },
                {
                    "id": "bioasq_test_4",
                    "body": "Describe the mechanism of metformin.",
                    "type": "summary",
                    "documents": [],
                    "snippets": [],
                    "ideal_answer": ["Metformin works by reducing hepatic glucose production."],
                },
            ]
        }
        
        file_path = temp_dir / "bioasq_test.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        return file_path

    def test_load_from_local(self, sample_bioasq_data: Path) -> None:
        """Test loading from local file."""
        loader = BioASQLoader(source="local", local_path=sample_bioasq_data)
        questions = loader.load()

        assert len(questions) == 4
        assert all(isinstance(q, BioASQQuestion) for q in questions)

    def test_question_types(self, sample_bioasq_data: Path) -> None:
        """Test parsing different question types."""
        loader = BioASQLoader(source="local", local_path=sample_bioasq_data)
        questions = loader.load()

        types = {q.question_type for q in questions}
        assert types == {"yesno", "factoid", "list", "summary"}

    def test_yesno_parsing(self, sample_bioasq_data: Path) -> None:
        """Test yes/no question parsing."""
        loader = BioASQLoader(source="local", local_path=sample_bioasq_data)
        questions = loader.load()

        yesno = next(q for q in questions if q.question_type == "yesno")
        assert yesno.exact_answer == "yes"
        assert len(yesno.gold_pmids) == 2
        assert "12345678" in yesno.gold_pmids

    def test_factoid_parsing(self, sample_bioasq_data: Path) -> None:
        """Test factoid question parsing."""
        loader = BioASQLoader(source="local", local_path=sample_bioasq_data)
        questions = loader.load()

        factoid = next(q for q in questions if q.question_type == "factoid")
        assert factoid.exact_answer == "BCR-ABL"

    def test_list_parsing(self, sample_bioasq_data: Path) -> None:
        """Test list question parsing."""
        loader = BioASQLoader(source="local", local_path=sample_bioasq_data)
        questions = loader.load()

        list_q = next(q for q in questions if q.question_type == "list")
        assert isinstance(list_q.exact_answer, list)
        assert "fever" in list_q.exact_answer

    def test_summary_parsing(self, sample_bioasq_data: Path) -> None:
        """Test summary question parsing."""
        loader = BioASQLoader(source="local", local_path=sample_bioasq_data)
        questions = loader.load()

        summary = next(q for q in questions if q.question_type == "summary")
        assert summary.ideal_answer is not None
        assert "metformin" in summary.ideal_answer.lower()

    def test_get_gold_pmids(self, sample_bioasq_data: Path) -> None:
        """Test getting gold PMIDs."""
        loader = BioASQLoader(source="local", local_path=sample_bioasq_data)
        pmids = loader.get_gold_pmids()

        assert "12345678" in pmids
        assert "87654321" in pmids
        assert "11111111" in pmids

    def test_sample_questions(self, sample_bioasq_data: Path) -> None:
        """Test sampling questions."""
        loader = BioASQLoader(source="local", local_path=sample_bioasq_data)
        sampled = loader.sample_questions(n=2, seed=42)

        assert len(sampled) == 2
        assert all(isinstance(q, BioASQQuestion) for q in sampled)

    def test_invalid_source(self) -> None:
        """Test invalid source raises error."""
        loader = BioASQLoader(source="invalid")
        with pytest.raises(ValueError, match="Unknown source"):
            loader.load()

    def test_missing_local_file(self) -> None:
        """Test missing local file raises error."""
        loader = BioASQLoader(source="local", local_path=Path("/nonexistent/file.json"))
        with pytest.raises(FileNotFoundError):
            loader.load()


class TestPubMedQALoader:
    """Tests for PubMedQALoader."""

    @pytest.fixture
    def sample_pubmedqa_data(self, temp_dir: Path) -> Path:
        """Create sample PubMedQA data file."""
        data = {
            "12345": {
                "question": "Does drug X improve patient outcomes?",
                "context": {
                    "contexts": [
                        "Background: Drug X has been studied extensively.",
                        "Methods: We conducted a randomized trial.",
                        "Results: Patients showed significant improvement.",
                    ]
                },
                "long_answer": "The study found that drug X significantly improves outcomes.",
                "final_decision": "yes",
            },
            "67890": {
                "question": "Is treatment Y safe for elderly patients?",
                "context": {
                    "contexts": [
                        "Safety concerns have been raised.",
                        "We analyzed adverse events.",
                    ]
                },
                "long_answer": "Treatment Y showed mixed safety results.",
                "final_decision": "maybe",
            },
            "11111": {
                "question": "Does intervention Z reduce mortality?",
                "context": {
                    "contexts": ["No significant effect was observed."]
                },
                "long_answer": "Intervention Z did not reduce mortality.",
                "final_decision": "no",
            },
        }

        file_path = temp_dir / "pubmedqa_test.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        return file_path

    def test_load_from_local(self, sample_pubmedqa_data: Path) -> None:
        """Test loading from local file."""
        loader = PubMedQALoader(source="local", local_path=sample_pubmedqa_data)
        questions = loader.load()

        assert len(questions) == 3
        assert all(isinstance(q, PubMedQAQuestion) for q in questions)

    def test_label_parsing(self, sample_pubmedqa_data: Path) -> None:
        """Test label parsing."""
        loader = PubMedQALoader(source="local", local_path=sample_pubmedqa_data)
        questions = loader.load()

        labels = {q.label for q in questions}
        assert labels == {"yes", "no", "maybe"}

    def test_context_parsing(self, sample_pubmedqa_data: Path) -> None:
        """Test context parsing."""
        loader = PubMedQALoader(source="local", local_path=sample_pubmedqa_data)
        questions = loader.load()

        q = next(q for q in questions if q.pmid == "12345")
        assert len(q.context) == 3
        assert "randomized trial" in q.context[1]

    def test_get_pmids(self, sample_pubmedqa_data: Path) -> None:
        """Test getting PMIDs."""
        loader = PubMedQALoader(source="local", local_path=sample_pubmedqa_data)
        pmids = loader.get_pmids()

        assert "12345" in pmids
        assert "67890" in pmids
        assert "11111" in pmids

    def test_get_label_distribution(self, sample_pubmedqa_data: Path) -> None:
        """Test label distribution."""
        loader = PubMedQALoader(source="local", local_path=sample_pubmedqa_data)
        dist = loader.get_label_distribution()

        assert dist["yes"] == 1
        assert dist["no"] == 1
        assert dist["maybe"] == 1

    def test_sample_questions(self, sample_pubmedqa_data: Path) -> None:
        """Test sampling questions."""
        loader = PubMedQALoader(source="local", local_path=sample_pubmedqa_data)
        sampled = loader.sample_questions(n=2, seed=42)

        assert len(sampled) == 2
        assert all(isinstance(q, PubMedQAQuestion) for q in sampled)

    def test_invalid_source(self) -> None:
        """Test invalid source raises error."""
        loader = PubMedQALoader(source="invalid")
        with pytest.raises(ValueError, match="Unknown source"):
            loader.load()

    def test_missing_local_file(self) -> None:
        """Test missing local file raises error."""
        loader = PubMedQALoader(source="local", local_path=Path("/nonexistent/file.json"))
        with pytest.raises(FileNotFoundError):
            loader.load()

