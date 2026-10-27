
import numpy as np
import pytest

from src.vector_store import VectorStore


@pytest.fixture
def sample_data():
    chunks = [
        {
            "text": "Antibiotic treatment guidance",
            "metadata": {"source_id": "guideline_a", "page_number": 2},
        },
        {
            "text": "Treatment duration guidance",
            "metadata": {"source_id": "guideline_b", "page_number": 5},
        },
        {
            "text": "Unrelated laboratory guidance",
            "metadata": {"source_id": "guideline_c", "page_number": 8},
        },
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return chunks, embeddings


def test_build_creates_index_and_stores_chunks(sample_data):
    chunks, embeddings = sample_data

    store = VectorStore()
    store.build(chunks, embeddings)

    assert store.index is not None
    assert store.index.ntotal == len(chunks)
    assert store.dimension == 3
    assert store.chunks == chunks


def test_search_returns_most_similar_chunk_first(sample_data):
    chunks, embeddings = sample_data

    store = VectorStore()
    store.build(chunks, embeddings)

    results = store.search(
        np.array([0.9, 0.1, 0.0], dtype=np.float32),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["text"] == chunks[0]["text"]
    assert results[0]["score"] > results[1]["score"]


def test_search_preserves_metadata_and_scores(sample_data):
    chunks, embeddings = sample_data

    store = VectorStore()
    store.build(chunks, embeddings)

    results = store.search(
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        top_k=1,
    )

    assert results[0]["metadata"] == chunks[1]["metadata"]
    assert isinstance(results[0]["score"], float)
    assert results[0]["score"] == pytest.approx(1.0)


def test_search_top_k_is_limited_to_available_chunks(sample_data):
    chunks, embeddings = sample_data

    store = VectorStore()
    store.build(chunks, embeddings)

    results = store.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=10,
    )

    assert len(results) == len(chunks)


def test_build_rejects_empty_chunks():
    store = VectorStore()

    with pytest.raises(ValueError):
        store.build([], np.empty((0, 3), dtype=np.float32))


def test_build_rejects_mismatched_chunk_and_embedding_counts(sample_data):
    chunks, embeddings = sample_data

    store = VectorStore()

    with pytest.raises(ValueError):
        store.build(chunks[:2], embeddings)


def test_search_before_build_fails():
    store = VectorStore()

    with pytest.raises(RuntimeError):
        store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_build_rejects_zero_embedding_vectors(sample_data):
    chunks, embeddings = sample_data
    embeddings[0] = [0.0, 0.0, 0.0]

    store = VectorStore()

    with pytest.raises(ValueError):
        store.build(chunks, embeddings)


def test_build_rejects_non_finite_embeddings(sample_data):
    chunks, embeddings = sample_data
    embeddings[0, 0] = np.nan

    store = VectorStore()

    with pytest.raises(ValueError):
        store.build(chunks, embeddings)


# Persistence tests

def test_save_and_load_preserves_search_results(tmp_path, sample_data):
    chunks, embeddings = sample_data

    original = VectorStore()
    original.build(chunks, embeddings)

    query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    before = original.search(query, top_k=2)

    original.save(tmp_path)
    restored = VectorStore.load(tmp_path)

    after = restored.search(query, top_k=2)

    assert len(after) == len(before)

    for old, new in zip(before, after):
        assert new["text"] == old["text"]
        assert new["metadata"] == old["metadata"]
        assert new["score"] == pytest.approx(old["score"])


def test_save_creates_both_files(tmp_path, sample_data):
    chunks, embeddings = sample_data

    store = VectorStore()
    store.build(chunks, embeddings)
    store.save(tmp_path)

    assert (tmp_path / "guidelines.faiss").is_file()
    assert (tmp_path / "chunks.json").is_file()


def test_save_before_build_fails(tmp_path):
    with pytest.raises(RuntimeError):
        VectorStore().save(tmp_path)


def test_load_missing_files_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path)
