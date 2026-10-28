
"""Compare query wording against the same guideline index."""

from src.retrieval_pipeline import build_registered_retriever

CASES = [
    {
        "id": "CG-02",
        "source": "nice_ng111",
        "page": 12,
        "queries": [
            (
                "Original",
                "When should NICE-recommended IV antibiotics be reviewed, "
                "and what should be considered at that review?",
            ),
            (
                "Guideline terminology",
                "NICE acute pyelonephritis: review intravenous antibiotics "
                "by 48 hours and consider stepping down to oral antibiotics.",
            ),
            (
                "Population-specific wording",
                "For pregnant women with acute pyelonephritis, when should "
                "intravenous antibiotics be reviewed and oral antibiotics "
                "considered?",
            ),
        ],
    },
    {
        "id": "CG-03",
        "source": "nice_ng111",
        "page": 7,
        "queries": [
            (
                "Original",
                "When urine culture results show resistance, how does NICE "
                "recommend changing antibiotic treatment?",
            ),
            (
                "Guideline terminology",
                "NICE acute pyelonephritis: change the antibiotic according "
                "to susceptibility results if the bacteria are resistant, "
                "using a narrow spectrum antibiotic wherever possible.",
            ),
            (
                "Culture-specific wording",
                "If urine culture and antibiotic susceptibility results show "
                "resistant bacteria in acute pyelonephritis, what does NICE "
                "recommend changing?",
            ),
        ],
    },
]


def main():
    retriever, report = build_registered_retriever()

    for case in CASES:
        print("\n" + "=" * 80)
        print(f"{case['id']} | Target: {case['source']} page {case['page']}")

        for label, query in case["queries"]:
            results = retriever.search(query, top_k=367)

            targets = [
                (rank, result)
                for rank, result in enumerate(results, start=1)
                if result.get("metadata", {}).get("source_id")
                == case["source"]
                and result.get("metadata", {}).get("page_number")
                == case["page"]
            ]

            print(f"\n{label}")
            print("Query:", query)

            if targets:
                for rank, result in targets:
                    print(
                        f"Target rank: {rank} | "
                        f"Score: {result.get('score')}"
                    )
            else:
                print("Target page was not returned.")

            print("Top result:")
            if results:
                first = results[0]
                metadata = first.get("metadata", {})
                print(
                    f"Source={metadata.get('source_id')} | "
                    f"Page={metadata.get('page_number')} | "
                    f"Score={first.get('score')}"
                )


if __name__ == "__main__":
    main()
