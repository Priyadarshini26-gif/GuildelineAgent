
"""Basic patient-context checks for ConstraintGuard-ED.

This module flags missing, invalid, or unresolved context.
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

# Basic input-format checks, not clinical decision rules.
YES_NO_FIELDS = {
    "allergies",
    "pregnancy_status",
    "oral_tolerance",
}

ACCEPTED_YES_NO = {
    "yes",
    "no",
    "unknown",
    "not known",
    "not applicable",
    "n/a",
    # Descriptive values used by the existing tests.
    "not pregnant",
    "no known medication allergies",
    "tolerating oral medication",
}

ACCEPTED_RENAL_VALUES = {
    "good",
    "normal",
    "impaired",
    "reduced",
    "unknown",
    "not known",
    "not assessed",
    "available for review",
}

ACCEPTED_SEVERITY_VALUES = {
    "mild",
    "moderate",
    "severe",
    "stable",
    "unstable",
    "unknown",
    "not known",
    "not assessed",
    "documented",
}

ACCEPTED_CULTURE_VALUES = {
    "pending",
    "available",
    "positive",
    "negative",
    "not collected",
    "not available",
    "unknown",
    "not known",
    "not assessed",
}


def _normalise(value: Any) -> str:
    """Convert a text value to a normalised form."""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _is_invalid(field: str, value: Any) -> bool:
    """Check whether a supplied value matches basic input rules."""

    if field == "age":
        # Reject booleans, which Python otherwise treats as numbers.
        if isinstance(value, bool):
            return True

        try:
            age = float(value)
            return not (0 < age < 125)
        except (TypeError, ValueError, OverflowError):
            return True

    normalised = _normalise(value)

    if not normalised:
        return False

    if field in YES_NO_FIELDS:
        return normalised not in ACCEPTED_YES_NO

    if field == "renal_function":
        return normalised not in ACCEPTED_RENAL_VALUES

    if field == "severity":
        return normalised not in ACCEPTED_SEVERITY_VALUES

    if field == "culture_results":
        return normalised not in ACCEPTED_CULTURE_VALUES

    return False


def _is_unresolved(field: str, value: Any) -> bool:
    """Flag recognised values that still leave information unresolved."""

    normalised = _normalise(value)

    if normalised in {
        "unknown",
        "not known",
        "not assessed",
    }:
        return True

    if field in YES_NO_FIELDS and normalised in {
        "unknown",
        "not known",
    }:
        return True

    return False


def verify_patient_constraints(
    patient_context: dict[str, Any],
    evidence_response: dict[str, Any],
) -> dict[str, Any]:
    """Identify missing, invalid, or unresolved patient context."""

    if not isinstance(patient_context, dict):
        raise TypeError("patient_context must be a dictionary.")

    if not isinstance(evidence_response, dict):
        raise TypeError("evidence_response must be a dictionary.")

    missing_context = []
    invalid_context = []
    unresolved_context = []

    for field, description in RELEVANT_CONTEXT_FIELDS.items():
        value = patient_context.get(field)

        # Case 1: The field is missing or blank.
        if value is None or (
            isinstance(value, str) and not value.strip()
        ):
            missing_context.append({
                "field": field,
                "description": description,
            })
            continue

        # Case 2: The field has an unrecognised or invalid value.
        if _is_invalid(field, value):
            invalid_context.append({
                "field": field,
                "description": description,
                "provided_value": str(value),
                "reason": (
                    "Value does not match an accepted input format."
                ),
            })
            continue

        # Case 3: The value is recognised but remains unresolved.
        if _is_unresolved(field, value):
            unresolved_context.append({
                "field": field,
                "description": description,
                "provided_value": str(value),
                "reason": (
                    "Value is unresolved and needs review."
                ),
            })

    evidence = evidence_response.get("evidence", [])

    if not isinstance(evidence, list):
        raise ValueError("Evidence must be a list.")

    usable_evidence = [
        item
        for item in evidence
        if isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and item["text"].strip()
    ]

    # Evidence availability takes priority over context status.
    if not usable_evidence:
        status = "insufficient_evidence"
    elif missing_context:
        status = "context_incomplete"
    elif invalid_context:
        status = "context_invalid_needs_review"
    elif unresolved_context:
        status = "context_unresolved_needs_review"
    else:
        status = "context_present_needs_review"

    return {
        "status": status,
        "missing_context": missing_context,
        "missing_context_count": len(missing_context),
        "invalid_context": invalid_context,
        "invalid_context_count": len(invalid_context),
        "unresolved_context": unresolved_context,
        "unresolved_context_count": len(unresolved_context),
        "evidence_count": len(usable_evidence),
        "clinical_validation_performed": False,
        "message": (
            "This check identifies missing, invalid, or unresolved "
            "information only. It does not establish treatment "
            "suitability or safety."
        ),
    }
