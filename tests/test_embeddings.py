
import numpy as np
import pytest

from src.embeddings import EmbeddingGenerator


class FakeModel:
    """A small fake model so unit tests do not download a real model."""

    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 3), dtype=np.float32)


@pytest.fixture
def generator():
    instance = EmbeddingGenerator.__new__(EmbeddingGenerator)
    instance.model = FakeModel()
    return instance


def test_embed_texts_returns_vectors(generator):
    result = generator.embed_texts(["Antibiotic guideline"])

    assert result.shape == (1, 3)


def test_embed_multiple_texts(generator):
    result = generator.embed_texts(["First passage", "Second passage"])

    assert result.shape == (2, 3)


def test_embed_empty_list(generator):
    assert generator.embed_texts([]) == []


def test_embed_texts_rejects_blank_text(generator):
    with pytest.raises(ValueError):
        generator.embed_texts([""])


def test_embed_query_returns_one_vector(generator):
    result = generator.embed_query("What does the guideline recommend?")

    assert result.shape == (3,)


def test_embed_query_rejects_blank_query(generator):
    with pytest.raises(ValueError):
        generator.embed_query("   ")
