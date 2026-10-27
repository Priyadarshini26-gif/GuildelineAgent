"""Evidence-grounded answer synthesis for ConstraintGuard-ED.

This first version is extractive: it presents retrieved guideline passages
with citations rather than inventing a clinical recommendation.
"""

from typing import Any


def synthesize_answer(evidence_response: dict[str, Any]) -> dict[str, Any]:
    """Format retrieved evidence into a structured, cited response."""

    if not isinstance(evidence_response, dict):
        raise TypeError("evidence_response must be a dictionary.")

    question = evidence_response.get("question", "")
    evidence = evidence_response.get("evidence", [])

    if not isinstance(question, str) or not question.strip():
        raise ValueError("The question must be a non-empty string.")

    if not isinstance(evidence, list):
        raise ValueError("Evidence must be provided as a list.")

    # No evidence means no clinical answer should be generated.
    usable_evidence = [
        item
        for item in evidence
        if isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and item["text"].strip()
    ]

    if not usable_evidence:
        return {
            "question": question,
            "status": "insufficient_evidence",
            "answer": (
                "I cannot provide an evidence-grounded answer because "
                "no usable guideline passages were retrieved."
            ),
            "citations": [],
            "limitations": [
                "No usable evidence was available.",
                "No clinical recommendation has been generated.",
            ],
        }

    citations = []
    answer_sections = []

    for number, item in enumerate(usable_evidence, start=1):
        source_id = item.get("source_id", "unknown")
        source_title = item.get("source_title", "Unknown source")
        page_number = item.get("page_number")
        chunk_id = item.get("chunk_id", "unknown")
        passage = item["text"].strip()

        citation = {
            "id": number,
            "source_id": source_id,
            "source_title": source_title,
            "page_number": page_number,
            "chunk_id": chunk_id,
        }
        citations.append(citation)

        page_label = (
            f"PDF page {page_number}"
            if page_number is not None
            else "PDF page unavailable"
        )

        answer_sections.append(
            f"[{number}] {passage}\n"
            f"Source: {source_title} ({source_id}), {page_label}."
        )

    answer = (
        "Retrieved guideline evidence:\n\n"
        + "\n\n".join(answer_sections)
    )

    limitations = [
        "Passages are presented as retrieved; they have not been "
        "independently validated for clinical applicability.",
        "The system has not yet verified patient-specific constraints.",
        "Different guidelines may address different populations or "
        "clinical circumstances. Do not assume their recommendations "
        "are interchangeable.",
    ]

    return {
        "question": question,
        "status": "answered",
        "answer": answer,
        "citations": citations,
        "limitations": limitations,
    }