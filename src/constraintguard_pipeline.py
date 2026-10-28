
"""End-to-end workflow for the ConstraintGuard-ED prototype."""

from typing import Any

from src.answer_synthesis import synthesize_answer
from src.constraint_verification import verify_patient_constraints
from src.retrieval_pipeline import build_registered_retriever
from src.risk_triggered_retrieval import retrieve_with_risk_trigger


def run_constraintguard(
    question: str,
    patient_context: dict[str, Any] | None = None,
    retriever=None,
) -> dict[str, Any]:
    """Run retrieval, context checks, and evidence-grounded formatting."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    context = patient_context or {}

    if not isinstance(context, dict):
        raise TypeError("patient_context must be a dictionary.")

    # Build the real retriever only when one was not supplied.
    if retriever is None:
        retriever, ingestion_report = build_registered_retriever()
    else:
        ingestion_report = "Existing retriever supplied by caller."

    # 1. Retrieve evidence, expanding when the policy triggers.
    evidence_response = retrieve_with_risk_trigger(
        retriever=retriever,
        question=question,
        patient_context=context,
        initial_top_k=3,
        expanded_top_k=8,
    )

    # 2. Identify missing patient context.
    constraint_report = verify_patient_constraints(
        patient_context=context,
        evidence_response=evidence_response,
    )

    # 3. Format retrieved passages with source/page citations.
    answer_response = synthesize_answer(evidence_response)

    # 4. Return one auditable record for future evaluation.
    return {
        "question": question.strip(),
        "patient_context": context,
        "retrieval": evidence_response,
        "constraint_report": constraint_report,
        "answer": answer_response,
        "ingestion_report": ingestion_report,
    }


def main():
    print("ConstraintGuard-ED — Research Prototype")
    print("Enter the clinical question you want to investigate.\n")

    question = input("Clinical question: ").strip()

    print(
        "\nEnter any patient information you have. "
        "Press Enter to leave a field unknown."
    )

    fields = [
        ("age", "Patient age"),
        ("allergies", "Medication allergies"),
        ("pregnancy_status", "Pregnancy status"),
        ("renal_function", "Renal function"),
        ("severity", "Illness severity"),
        ("oral_tolerance", "Ability to take oral medication"),
        ("culture_results", "Culture and susceptibility results"),
    ]

    patient_context = {}

    for field, label in fields:
        value = input(f"{label}: ").strip()
        if value:
            patient_context[field] = value

    result = run_constraintguard(question, patient_context)

    print("\n" + "=" * 60)
    print("CONSTRAINTGUARD-ED RESULTS")
    print("=" * 60)

    print("\nRETRIEVAL STATUS:", result["retrieval"]["status"])
    print("Expansion triggered:", result["retrieval"]["expansion_triggered"])
    print("Expansion reasons:", result["retrieval"]["expansion_reasons"])
    print("Evidence passages:", len(result["retrieval"]["evidence"]))

    print("\nPATIENT-CONTEXT CHECK:", result["constraint_report"]["status"])

    missing = result["constraint_report"]["missing_context"]
    if missing:
        print("Missing context:")
        for item in missing:
            print("-", item["description"])
    else:
        print("No required context fields were left blank.")

    print("\nEVIDENCE-BOUNDED RESPONSE")
    print(result["answer"]["answer"])

    print("\nLIMITATIONS")
    for item in result["answer"]["limitations"]:
        print("-", item)

    print("\nRETRIEVAL LIMITATIONS")
    for item in result["retrieval"]["limitations"]:
        print("-", item)

    print(
        "\nResearch prototype only: this output is not a validated "
        "clinical recommendation or a substitute for clinician review."
    )


if __name__ == "__main__":
    main()
