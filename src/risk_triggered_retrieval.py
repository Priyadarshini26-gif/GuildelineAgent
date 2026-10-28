
"""Risk-triggered retrieval prototype for ConstraintGuard-ED."""

from typing import Any


RISK_TERMS = {
    "allergy": "medication allergy",
    "allergic": "medication allergy",
    "pregnan": "pregnancy",
    "renal": "renal function",
    "kidney": "renal function",
    "sepsis": "severe infection",
    "severe": "illness severity",
    "resistan": "antimicrobial resistance",
    "contraindication": "contraindications",
    "interaction": "drug interactions",
    "culture": "culture and susceptibility",
    "susceptibility": "culture and susceptibility",
    "child": "age-specific recommendations",
    "paediatric": "age-specific recommendations",
    "pediatric": "age-specific recommendations",
    "dosage": "dose safety",
    "dose": "dose safety",
    "intravenous": "route of administration",
    "iv antibiotic": "route of administration",
}


def identify_risk_triggers(question: str) -> list[str]:
    """Return predefined risk categories mentioned in a question."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    normalized = question.lower()
    triggers = {
        category
        for term, category in RISK_TERMS.items()
        if term in normalized
    }
    return sorted(triggers)


def _evidence_key(item: dict[str, Any]) -> str:
    """Create a stable key for removing duplicate retrieved chunks."""

    chunk_id = item.get("chunk_id")
    if chunk_id:
        return str(chunk_id)

    metadata = item.get("metadata", {})
    return "|".join(
        str(value)
        for value in (
            metadata.get("source_id", ""),
            metadata.get("page_number", ""),
            item.get("text", ""),
        )
    )


def retrieve_with_risk_trigger(
    retriever,
    question: str,
    patient_context: dict[str, Any] | None = None,
    initial_top_k: int = 3,
    expanded_top_k: int = 8,
) -> dict[str, Any]:
    """Retrieve initial evidence and expand retrieval when triggered."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if initial_top_k < 1 or expanded_top_k < initial_top_k:
        raise ValueError(
            "Use initial_top_k >= 1 and expanded_top_k >= initial_top_k."
        )

    context = patient_context or {}
    if not isinstance(context, dict):
        raise TypeError("patient_context must be a dictionary.")

    clean_question = question.strip()
    triggers = identify_risk_triggers(clean_question)

    required_fields = (
        "age",
        "allergies",
        "pregnancy_status",
        "renal_function",
        "severity",
        "oral_tolerance",
        "culture_results",
    )

    missing_context = [
        field
        for field in required_fields
        if context.get(field) is None
        or (
            isinstance(context.get(field), str)
            and not context[field].strip()
        )
    ]

    initial_results = retriever.search(
        clean_question,
        top_k=initial_top_k,
    )

    usable_initial = [
        item for item in initial_results
        if isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and item["text"].strip()
    ]

    reasons = []

    if triggers:
        reasons.append("risk_terms_detected")

    if missing_context:
        reasons.append("patient_context_missing")

    if not usable_initial:
        reasons.append("initial_retrieval_empty")

    should_expand = bool(reasons)

    expanded_results = []
    if should_expand:
        # Expand the retrieval scope using the same question.
        # This is a wider retrieval pass, not a new clinical query.
        expanded_results = retriever.search(
            clean_question,
            top_k=expanded_top_k,
        )

    combined = []
    seen = set()

    # Keep the initial results first, then append new results.
    for item in usable_initial + expanded_results:
        if not isinstance(item, dict):
            continue
        if not isinstance(item.get("text"), str) or not item["text"].strip():
            continue

        key = _evidence_key(item)
        if key not in seen:
            seen.add(key)
            combined.append(item)

    return {
        "question": clean_question,
        "status": (
            "evidence_retrieved"
            if combined
            else "no_evidence_retrieved"
        ),
        "risk_triggers": triggers,
        "missing_context": missing_context,
        "expansion_triggered": should_expand,
        "expansion_reasons": reasons,
        "initial_evidence_count": len(usable_initial),
        "expanded_evidence_count": len(expanded_results),
        "evidence": combined,
        "limitations": [
            "Risk triggers use predefined keywords and are not a "
            "validated clinical risk classifier.",
            "Expanded retrieval uses the same question and a larger "
            "result count; it does not guarantee better evidence.",
            "Missing context flags do not establish whether a treatment "
            "is appropriate or safe.",
        ],
    }
