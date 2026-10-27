
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_pipeline import build_registered_retriever


INDEX_DIR = PROJECT_ROOT / "data" / "index"


def main():
    print("Building the guideline retrieval index...")

    retriever, report = build_registered_retriever()

    print("\nGuideline ingestion report:")
    print(report)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    retriever.vector_store.save(INDEX_DIR)

    print("\nIndex saved successfully.")
    print(f"Index directory: {INDEX_DIR}")
    print(f"FAISS file: {INDEX_DIR / 'guidelines.faiss'}")
    print(f"Chunk metadata: {INDEX_DIR / 'chunks.json'}")


if __name__ == "__main__":
    main()
