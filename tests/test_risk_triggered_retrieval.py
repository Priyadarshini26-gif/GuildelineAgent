
import pytest

from src.risk_triggered_retrieval import (
    identify_risk_triggers,
    retrieve_with_risk_trigger,
)


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, question, top_k=5):
        self.calls.append({"question": question, "top_k": top_k})
        return self.results[:top_k]


def make_evidence(chunk_id, text="Example guideline passage."):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "source_id": "test_source",
            "page_number": 1,
        },
    }


def test_identifies_predefined_risk_terms():
    triggers = identify_risk_triggers(
        "Is this antibiotic appropriate for a patient with renal impairment?"
    )

    assert "renal function" in triggers


def test_expands_retrieval_when_risk_term_is_detected():
    retriever = FakeRetriever([
        make_evidence("chunk_1"),
        make_evidence("chunk_2"),
        make_evidence("chunk_3"),
        make_evidence("chunk_4"),
    ])

    result = retrieve_with_risk_trigger(
        retriever,
        "Which antibiotic is appropriate for a patient with a drug allergy?",
        patient_context={
            "age": 45,
            "allergies": "Penicillin allergy",
            "pregnancy_status": "Not pregnant",
            "renal_function": "Available",
            "severity": "Documented",
            "oral_tolerance": "Tolerating oral medication",
            "culture_results": "Pending",
        },
    )

    assert result["expansion_triggered"] is True
    assert "risk_terms_detected" in result["expansion_reasons"]
    assert len(retriever.calls) == 2
    assert retriever.calls[1]["top_k"] == 8


def test_expands_when_patient_context_is_missing():
    retriever = FakeRetriever([make_evidence("chunk_1")])

    result = retrieve_with_risk_trigger(
        retriever,
        "Explain the guideline recommendation.",
        patient_context={},
    )

    assert result["expansion_triggered"] is True
    assert "patient_context_missing" in result["expansion_reasons"]
    assert "age" in result["missing_context"]


def test_deduplicates_evidence_after_expansion():
    retriever = FakeRetriever([
        make_evidence("chunk_1"),
        make_evidence("chunk_2"),
        make_evidence("chunk_3"),
        make_evidence("chunk_4"),
    ])

    result = retrieve_with_risk_trigger(
        retriever,
        "What should be considered for a patient with sepsis?",
        patient_context={},
    )

    ids = [item["chunk_id"] for item in result["evidence"]]
    assert len(ids) == len(set(ids))


def test_rejects_empty_question():
    retriever = FakeRetriever([])

    with pytest.raises(ValueError):
        retrieve_with_risk_trigger(retriever, "   ")
