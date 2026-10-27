
from src.retrieval_pipeline import build_registered_retriever


def main():
    retriever, report = build_registered_retriever()

    print("\nIngestion report:")
    for item in report:
        print(item)

    question = input("\nEnter a guideline question: ").strip()

    results = retriever.search(question, top_k=5)

    print("\nRetrieved passages:")
    for number, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"\n--- Result {number} ---")
        print(f"Source: {metadata.get('source_title')}")
        print(f"Source ID: {metadata.get('source_id')}")
        print(f"Page: {metadata.get('page_number')}")
        print(f"Similarity: {result['score']:.3f}")
        print(result["text"])


if __name__ == "__main__":
    main()
