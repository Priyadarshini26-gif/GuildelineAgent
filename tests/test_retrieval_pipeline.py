
import numpy as np
import pytest

from src.retrieval_pipeline import GuidelineRetriever


class FakeEmbeddingGenerator:
    """Deterministic embeddings for testing the connected pipeline."""

    def embed_texts(self, texts):
        vectors = []

        for text in texts:
            if "antibiotic" in text.lower():
                vectors.append([1.0, 0.0, 0.0])
            elif "duration" in text.lower():
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])

        return np.array(vectors, dtype=np.float32)

    def embed_query(self, question):
        if "antibiotic" in question.lower():
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "duration" in question.lower():
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)

        return np.array([0.0, 0.0, 1.0], dtype=np.float32)


@pytest.fixture
def chunks():
    return [
        {
            "text": "Antibiotic treatment guidance",
            "metadata": {"source_id": "guideline_a", "page_number": 2},
        },
        {
            "text": "Recommended treatment duration",
            "metadata": {"source_id": "guideline_b", "page_number": 5},
        },
        {
            "text": "General infection prevention",
            "metadata": {"source_id": "guideline_c", "page_number": 7},
        },
    ]


@pytest.fixture
def retriever():
    return GuidelineRetriever(
        embedding_generator=FakeEmbeddingGenerator()
    )


def test_build_and_search(retriever, chunks):
    retriever.build(chunks)

    results = retriever.search("antibiotic treatment", top_k=1)

    assert len(results) == 1
    assert results[0]["metadata"]["source_id"] == "guideline_a"


def test_search_preserves_source_and_page(retriever, chunks):
    retriever.build(chunks)

    results = retriever.search("duration", top_k=1)

    assert results[0]["metadata"]["source_id"] == "guideline_b"
    assert results[0]["metadata"]["page_number"] == 5


def test_search_before_build_fails(retriever):
    with pytest.raises(RuntimeError):
        retriever.search("antibiotic treatment")


def test_empty_chunks_are_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.build([])


def test_blank_question_is_rejected(retriever, chunks):
    retriever.build(chunks)

    with pytest.raises(ValueError):
        retriever.search("   ")
