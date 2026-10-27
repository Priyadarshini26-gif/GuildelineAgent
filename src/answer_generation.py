
"""Evidence-grounded response preparation for ConstraintGuard-ED."""
from src.answer_synthesis import synthesize_answer
from src.retrieval_pipeline import build_registered_retriever


def prepare_evidence_response(
    retriever,
    question: str,
    top_k: int = 5,
) -> dict:
    """Retrieve guideline evidence and return it with source provenance."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    results = retriever.search(question.strip(), top_k=top_k)

    evidence = []

    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})

        evidence.append({
            "rank": rank,
            "text": result.get("text", ""),
            "source_id": metadata.get("source_id", "unknown"),
            "source_title": metadata.get("source_title", "Unknown source"),
            "page_number": metadata.get("page_number"),
            "chunk_id": result.get("chunk_id", "unknown"),
        })

    if not evidence:
        return {
            "question": question,
            "status": "no_evidence_retrieved",
            "message": (
                "No guideline evidence was retrieved. "
                "Do not generate a clinical recommendation from this result."
            ),
            "evidence": [],
        }

    return {
        "question": question,
        "status": "evidence_retrieved",
        "message": (
            "Retrieved passages are supporting evidence, not a validated "
            "clinical recommendation. Check relevance and patient context."
        ),
        "evidence": evidence,
    }


def main():
    retriever, report = build_registered_retriever()

    print("Guideline ingestion report:")
    print(report)

    question = input("\nEnter a clinical guideline question: ").strip()

    response = prepare_evidence_response(retriever, question)
    if response["status"] == "evidence_retrieved":
        final_response = synthesize_answer(response)
    else:
        final_response = {
            "status": response["status"],
            "answer": response["message"],
            "limitations": ["No guideline evidence was available to support an answer."],
        }

    print("\n" + "=" * 60)
    print("CONSTRAINTGUARD-ED: EVIDENCE-GROUNDED RESPONSE")
    print("=" * 60)

    print("\nSTATUS:", final_response["status"])
    print("\nANSWER:\n", final_response["answer"])

    print("\nLIMITATIONS:")
    for limitation in final_response["limitations"]:
        print("-", limitation)

    print("\nSTATUS:", response["status"])
    print(response["message"])

    for item in response["evidence"]:
        print(f"\nEvidence {item['rank']}")
        print("Source:", item["source_title"])
        print("Source ID:", item["source_id"])
        print("PDF page:", item["page_number"])
        print("Passage:", item["text"])


if __name__ == "__main__":
    main()
