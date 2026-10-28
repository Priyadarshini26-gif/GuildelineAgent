
"""Inspect retrieved source pages for CG-02 and CG-03."""

from src.retrieval_pipeline import build_registered_retriever
from scripts.compare_retrieval import load_cases, get_source_page

TARGET_IDS = {"CG-02", "CG-03"}
TOP_K = 8


def main():
    retriever, report = build_registered_retriever()
    cases = [
        case for case in load_cases()
        if case["id"] in TARGET_IDS
    ]

    print("INDEX:", report)

    for case in cases:
        print("\n" + "=" * 72)
        print(case["id"], "-", case["question"])
        print("Expected source:", case["source_id"])
        print("Expected page(s):", case["pages"])

        results = retriever.search(
            case["question"],
            top_k=TOP_K,
        )

        for rank, item in enumerate(results, start=1):
            source_id, page = get_source_page(item)
            metadata = item.get("metadata", {})
            content = (
                item.get("text")
                or item.get("page_content")
                or metadata.get("text")
                or ""
            )

            print(f"\nRank {rank}")
            print("Source:", source_id)
            print("Page:", page)
            print("Text:", str(content).replace("\n", " ")[:700])


if __name__ == "__main__":
    main()
