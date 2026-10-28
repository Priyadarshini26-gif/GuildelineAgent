
"""Diagnostic TF-IDF retrieval baseline; does not modify the main pipeline."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.ingest_guidelines import ingest_registered_guidelines


CASES = [
    {
        "id": "CG-02-original",
        "question": (
            "When should NICE-recommended IV antibiotics be reviewed, "
            "and what should be considered at that review?"
        ),
        "source": "nice_ng111",
        "page": 12,
    },
    {
        "id": "CG-02-guideline-terms",
        "question": (
            "NICE acute pyelonephritis: review intravenous antibiotics "
            "by 48 hours and consider stepping down to oral antibiotics."
        ),
        "source": "nice_ng111",
        "page": 12,
    },
    {
        "id": "CG-03-original",
        "question": (
            "When urine culture results show resistance, how does NICE "
            "recommend changing antibiotic treatment?"
        ),
        "source": "nice_ng111",
        "page": 7,
    },
    {
        "id": "CG-03-guideline-terms",
        "question": (
            "NICE acute pyelonephritis: change the antibiotic according "
            "to susceptibility results if the bacteria are resistant, "
            "using a narrow spectrum antibiotic wherever possible."
        ),
        "source": "nice_ng111",
        "page": 7,
    },
]


def main():
    chunks, report = ingest_registered_guidelines()
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)

    print(f"Indexed chunks: {len(chunks)}")

    for case in CASES:
        query_vector = vectorizer.transform([case["question"]])
        scores = cosine_similarity(query_vector, matrix).ravel()
        ranked_indices = scores.argsort()[::-1]

        target_ranks = [
            rank
            for rank, idx in enumerate(ranked_indices, start=1)
            if (
                chunks[idx]["metadata"]["source_id"] == case["source"]
                and chunks[idx]["metadata"]["page_number"] == case["page"]
            )
        ]

        print("\n" + "=" * 70)
        print(f"{case['id']} — expected page {case['page']}")
        print(f"Question: {case['question']}")

        if target_ranks:
            print("Expected-page chunk ranks:", target_ranks)
        else:
            print("Expected page not found.")

        print("Top 5 lexical results:")

        for rank, idx in enumerate(ranked_indices[:5], start=1):
            chunk = chunks[idx]
            metadata = chunk["metadata"]

            print(
                f"\nRank {rank} | Score {scores[idx]:.4f} | "
                f"{metadata['source_id']} | "
                f"Page {metadata['page_number']} | "
                f"Chunk {metadata['chunk_number']}"
            )
            print(chunk["text"][:400].replace("\n", " "))


if __name__ == "__main__":
    main()
