
import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """Store embeddings, retrieve chunks, and persist the index."""

    def __init__(self):
        self.index = None
        self.chunks = []
        self.dimension = None

    def build(self, chunks, embeddings):
        """Build an exact similarity index from normalized embeddings."""
        if not chunks:
            raise ValueError("Cannot build an index without chunks.")

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have one embedding.")

        if embeddings.shape[1] == 0:
            raise ValueError("Embedding dimension must be positive.")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings must contain only finite values.")

        if np.any(np.linalg.norm(embeddings, axis=1) == 0):
            raise ValueError("Embeddings must not be zero vectors.")

        for chunk in chunks:
            if (
                not isinstance(chunk, dict)
                or not isinstance(chunk.get("text"), str)
                or not isinstance(chunk.get("metadata"), dict)
            ):
                raise ValueError(
                    "Each chunk must contain text and metadata."
                )

        faiss.normalize_L2(embeddings)

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.chunks = list(chunks)

    def search(self, query_embedding, top_k=5, source_id=None):
        """Return the most similar chunks, optionally from one source."""
        if self.index is None:
            raise RuntimeError("Build or load the index before searching.")

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer.")

        if source_id is not None and (
            not isinstance(source_id, str) or not source_id.strip()
        ):
            raise ValueError(
                "source_id must be a non-empty string or None."
            )

        query_embedding = np.asarray(
            query_embedding, dtype=np.float32
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape != (1, self.dimension):
            raise ValueError("Query embedding has the wrong dimensions.")

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding must contain finite values.")

        if np.linalg.norm(query_embedding) == 0:
            raise ValueError("Query embedding must not be a zero vector.")

        faiss.normalize_L2(query_embedding)

        # Rank all chunks before applying the optional source filter.
        scores, indices = self.index.search(
            query_embedding, len(self.chunks)
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            chunk = self.chunks[index]
            metadata = chunk["metadata"]

            if (
                source_id is not None
                and metadata.get("source_id") != source_id
            ):
                continue

            results.append(
                {
                    "text": chunk["text"],
                    "metadata": metadata,
                    "score": float(score),
                }
            )

            if len(results) >= top_k:
                break

        return results

    def save(self, directory):
        """Save the FAISS index and matching chunk metadata."""
        if self.index is None:
            raise RuntimeError("Build the index before saving.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "guidelines.faiss"
        metadata_path = directory / "chunks.json"

        faiss.write_index(self.index, str(index_path))

        metadata = {
            "dimension": self.dimension,
            "chunks": self.chunks,
        }

        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory):
        """Load a saved index and its matching metadata."""
        directory = Path(directory)
        index_path = directory / "guidelines.faiss"
        metadata_path = directory / "chunks.json"

        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                "Saved index or chunk metadata was not found."
            )

        index = faiss.read_index(str(index_path))
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        chunks = metadata.get("chunks")
        dimension = metadata.get("dimension")

        if not isinstance(chunks, list) or not chunks:
            raise ValueError("Saved chunk metadata is empty or invalid.")

        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 1
        ):
            raise ValueError("Saved embedding dimension is invalid.")

        if index.d != dimension:
            raise ValueError("Index and metadata dimensions do not match.")

        if index.ntotal != len(chunks):
            raise ValueError("Index and chunk counts do not match.")

        for chunk in chunks:
            if (
                not isinstance(chunk, dict)
                or not isinstance(chunk.get("text"), str)
                or not isinstance(chunk.get("metadata"), dict)
            ):
                raise ValueError("Saved chunk metadata has an invalid format.")

        store = cls()
        store.index = index
        store.chunks = chunks
        store.dimension = dimension

        return store
