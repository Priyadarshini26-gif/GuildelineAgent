
import pytest

from src.constraintguard_pipeline import run_constraintguard


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, question, top_k=5):
        self.calls.append((question, top_k))
        return self.results[:top_k]


def make_evidence(chunk_id, text):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "source_id": "test_guideline",
            "source_title": "Test Guideline",
            "page_number": 4,
        },
    }


def test_end_to_end_returns_evidence_citations_and_context_report():
    retriever = FakeRetriever([
        make_evidence("chunk_1", "Example guideline passage."),
    ])

    result = run_constraintguard(
        "Explain the guideline recommendation.",
        patient_context={},
        retriever=retriever,
    )

    assert result["retrieval"]["status"] == "evidence_retrieved"
    assert result["constraint_report"]["status"] == "context_incomplete"
    assert result["answer"]["status"] == "answered"
    assert result["answer"]["citations"][0]["source_id"] == "test_guideline"


def test_end_to_end_expands_retrieval_for_risk_trigger():
    retriever = FakeRetriever([
        make_evidence("chunk_1", "Example guideline passage."),
        make_evidence("chunk_2", "Another example passage."),
        make_evidence("chunk_3", "A third example passage."),
    ])

    context = {
        "age": 45,
        "allergies": "Penicillin allergy",
        "pregnancy_status": "Not pregnant",
        "renal_function": "Available",
        "severity": "Documented",
        "oral_tolerance": "Tolerating oral medication",
        "culture_results": "Pending",
    }

    result = run_constraintguard(
        "What should be considered for a patient with a drug allergy?",
        patient_context=context,
        retriever=retriever,
    )

    assert result["retrieval"]["expansion_triggered"] is True
    assert "medication allergy" in result["retrieval"]["risk_triggers"]
    assert len(retriever.calls) == 2


def test_end_to_end_abstains_when_no_evidence_is_returned():
    retriever = FakeRetriever([])

    result = run_constraintguard(
        "Explain the guideline recommendation.",
        patient_context={},
        retriever=retriever,
    )

    assert result["retrieval"]["status"] == "no_evidence_retrieved"
    assert result["answer"]["status"] == "insufficient_evidence"
    assert result["answer"]["citations"] == []


def test_end_to_end_rejects_empty_question():
    with pytest.raises(ValueError):
        run_constraintguard("   ", retriever=FakeRetriever([]))
