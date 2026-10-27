
"""Basic patient-context checks for ConstraintGuard-ED.

This module flags missing or potentially relevant context.
It does not determine whether a treatment is clinically safe.
"""

from typing import Any


RELEVANT_CONTEXT_FIELDS = {
    "age": "Patient age",
    "allergies": "Medication allergies",
    "pregnancy_status": "Pregnancy status",
    "renal_function": "Renal function",
    "severity": "Illness severity",
    "oral_tolerance": "Ability to take oral medication",
    "culture_results": "Culture and susceptibility results",
}


def verify_patient_constraints(
    patient_context: dict[str, Any],
    evidence_response: dict[str, Any],
) -> dict[str, Any]:
    """Identify missing context and return evidence for review."""

    if not isinstance(patient_context, dict):
        raise TypeError("patient_context must be a dictionary.")

    if not isinstance(evidence_response, dict):
        raise TypeError("evidence_response must be a dictionary.")

    missing_context = []

    for field, description in RELEVANT_CONTEXT_FIELDS.items():
        value = patient_context.get(field)

        if value is None or (
            isinstance(value, str) and not value.strip()
        ):
            missing_context.append({
                "field": field,
                "description": description,
            })

    evidence = evidence_response.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("Evidence must be a list.")

    usable_evidence = [
        item for item in evidence
        if isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and item["text"].strip()
    ]

    if not usable_evidence:
        status = "insufficient_evidence"
    elif missing_context:
        status = "context_incomplete"
    else:
        status = "context_present_needs_review"

    return {
        "status": status,
        "missing_context": missing_context,
        "missing_context_count": len(missing_context),
        "evidence_count": len(usable_evidence),
        "clinical_validation_performed": False,
        "message": (
            "This check identifies missing information only. "
            "It does not establish treatment suitability or safety."
        ),
    }
