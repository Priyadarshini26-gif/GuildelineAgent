
"""Inspect similarity ranking for the two NICE evaluation cases."""

from src.retrieval_pipeline import build_registered_retriever

CASES = [
    {
        "id": "CG-02",
        "question": (
            "When should NICE-recommended IV antibiotics be reviewed, "
            "and what should be considered at that review?"
        ),
        "source_id": "nice_ng111",
        "expected_page": 12,
    },
    {
        "id": "CG-03",
        "question": (
            "When urine culture results show resistance, how does NICE "
            "recommend changing antibiotic treatment?"
        ),
        "source_id": "nice_ng111",
        "expected_page": 7,
    },
]


def main():
    retriever, report = build_registered_retriever()

    for case in CASES:
        print("\n" + "=" * 80)
        print(f"{case['id']}: {case['question']}")
        print(
            f"Expected target: {case['source_id']}, "
            f"physical page {case['expected_page']}"
        )

        # Retrieve all chunks so we can calculate the target's actual rank.
        results = retriever.search(case["question"], top_k=367)

        target_results = [
            (rank, result)
            for rank, result in enumerate(results, start=1)
            if result.get("metadata", {}).get("source_id")
            == case["source_id"]
            and result.get("metadata", {}).get("page_number")
            == case["expected_page"]
        ]

        if target_results:
            print("\nExpected-page chunks and ranks:")
            for rank, result in target_results:
                print(
                    f"Rank {rank} | "
                    f"Score: {result.get('score')} | "
                    f"Chunk: {result.get('metadata', {}).get('chunk_number')}"
                )
                print(result.get("text", "")[:500])
        else:
            print("\nWARNING: No expected-page chunk found in search results.")

        print("\nTop 8 retrieved chunks:")
        for rank, result in enumerate(results[:8], start=1):
            metadata = result.get("metadata", {})
            print(
                f"\nRank {rank} | Score: {result.get('score')} | "
                f"Source: {metadata.get('source_id')} | "
                f"Page: {metadata.get('page_number')} | "
                f"Chunk: {metadata.get('chunk_number')}"
            )
            print(result.get("text", "")[:350])


if __name__ == "__main__":
    main()
