
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import EmbeddingGenerator
from src.retrieval_pipeline import GuidelineRetriever
from src.vector_store import VectorStore


INDEX_DIR = PROJECT_ROOT / "data" / "index"


def main():
    print("Loading saved guideline index...")

    # Load existing vectors and chunks from disk.
    vector_store = VectorStore.load(INDEX_DIR)

    # Load the model for embedding new questions only.
    embedding_generator = EmbeddingGenerator()

    retriever = GuidelineRetriever(
        embedding_generator=embedding_generator,
        vector_store=vector_store,
    )
    retriever.is_built = True

    print(f"Loaded chunks: {len(vector_store.chunks)}")
    print(f"Embedding dimension: {vector_store.dimension}")

    question = input("\nEnter your guideline question: ").strip()

    if not question:
        print("Please enter a non-empty question.")
        return

    results = retriever.search(question, top_k=3)

    print("\nTop retrieved passages")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"\nResult {rank}")
        print(f"Source: {metadata.get('source_id', 'Unknown')}")
        print(f"Page: {metadata.get('page_number', 'Unknown')}")
        print(f"Similarity score: {result['score']:.4f}")
        print(f"Text: {result['text'][:700]}")


if __name__ == "__main__":
    main()
