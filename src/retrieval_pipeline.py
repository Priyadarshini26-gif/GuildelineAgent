
from src.embeddings import EmbeddingGenerator
from src.ingest_guidelines import ingest_registered_guidelines
from src.vector_store import VectorStore


class GuidelineRetriever:
    """Connect guideline chunks, embeddings, and semantic search."""

    def __init__(self, embedding_generator=None, vector_store=None):
        self.embedding_generator = (
            embedding_generator or EmbeddingGenerator()
        )
        self.vector_store = vector_store or VectorStore()
        self.is_built = False

    def build(self, chunks):
        """Embed chunks and add them to the vector index."""
        if not chunks:
            raise ValueError("Cannot build retrieval from empty chunks.")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_generator.embed_texts(texts)

        self.vector_store.build(chunks, embeddings)
        self.is_built = True

    def search(self, question, top_k=5, source_id=None):
        """Find relevant passages, optionally from one guideline."""
        if not self.is_built:
            raise RuntimeError("Build the retriever before searching.")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        query_embedding = self.embedding_generator.embed_query(question)

        return self.vector_store.search(
            query_embedding,
            top_k=top_k,
            source_id=source_id,
        )


def build_registered_retriever():
    """Ingest registered PDFs and build a real embedding index."""
    chunks, report = ingest_registered_guidelines()

    if not chunks:
        raise RuntimeError(
            "No eligible guideline chunks were available for retrieval."
        )

    retriever = GuidelineRetriever()
    retriever.build(chunks)

    return retriever, report
