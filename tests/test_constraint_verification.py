
from src.constraint_verification import verify_patient_constraints


def test_flags_missing_patient_context():
    result = verify_patient_constraints(
        patient_context={"age": 45},
        evidence_response={
            "evidence": [
                {"text": "Example guideline passage."}
            ]
        },
    )

    assert result["status"] == "context_incomplete"
    assert result["missing_context_count"] == 6
    assert result["clinical_validation_performed"] is False


def test_reports_insufficient_evidence():
    result = verify_patient_constraints(
        patient_context={},
        evidence_response={"evidence": []},
    )

    assert result["status"] == "insufficient_evidence"
    assert result["evidence_count"] == 0


def test_complete_context_still_requires_review():
    context = {
        "age": 45,
        "allergies": "No known medication allergies",
        "pregnancy_status": "Not pregnant",
        "renal_function": "Available for review",
        "severity": "Documented",
        "oral_tolerance": "Tolerating oral medication",
        "culture_results": "Pending",
    }

    result = verify_patient_constraints(
        patient_context=context,
        evidence_response={
            "evidence": [
                {"text": "Example guideline passage."}
            ]
        },
    )

    assert result["status"] == "context_present_needs_review"
    assert result["missing_context_count"] == 0
    assert result["clinical_validation_performed"] is False
