"""Compare dense, TF-IDF, and hybrid retrieval.

Primary evaluation:
    The five original questions in data/evaluation/constraintguard_cases.json.

Diagnostic evaluation:
    Separate wording variants for CG-02 and CG-03.

Query-expansion evaluation:
    Additional manually expanded queries for CG-02 and CG-03.

This is a retrieval-only experiment. It does not modify the main pipeline
or establish clinical correctness or patient safety.
"""

import json
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.ingest_guidelines import ingest_registered_guidelines
from src.retrieval_pipeline import build_registered_retriever


CASES_FILE = Path("data/evaluation/constraintguard_cases.json")

TOP_K = 8
METRIC_K = 5
RRF_K = 60
PREVIEW_LENGTH = 500


# Existing alternate-wording diagnostic questions.
DIAGNOSTIC_QUESTIONS = {
    "CG-02": (
        "In pregnant women with acute pyelonephritis, when should "
        "NICE-recommended intravenous antibiotics be reviewed, "
        "and what should be considered at that review?"
    ),
    "CG-03": (
        "According to NICE, if susceptibility results show that "
        "the bacteria causing acute pyelonephritis are resistant "
        "to the current antibiotic, what change in treatment "
        "is recommended?"
    ),
}


# New query-expansion probes.
# These are diagnostic only and do not count toward the primary score.
EXPANDED_QUESTIONS = {
    "CG-02": (
        "NICE NG111 pregnant women acute pyelonephritis "
        "review intravenous antibiotics by 48 hours "
        "consider stepping down to oral antibiotics"
    ),
    "CG-03": (
        "NICE NG111 acute pyelonephritis change antibiotic "
        "according to urine culture susceptibility results "
        "if bacteria are resistant use a narrow spectrum antibiotic"
    ),
}


def get_field(obj, key, default=None):
    """Read a field from a dictionary or an object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def get_text(chunk):
    """Extract text from common chunk/result formats."""
    text = get_field(chunk, "text")
    if isinstance(text, str):
        return text

    page_content = get_field(chunk, "page_content")
    if isinstance(page_content, str):
        return page_content

    document = get_field(chunk, "document")
    if document is not None:
        document_text = get_field(document, "page_content")
        if isinstance(document_text, str):
            return document_text

    return ""


def get_metadata(chunk):
    """Extract metadata from common result formats."""
    metadata = get_field(chunk, "metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    if not metadata:
        document = get_field(chunk, "document")
        if document is not None:
            document_metadata = get_field(document, "metadata", {})
            if isinstance(document_metadata, dict):
                metadata = document_metadata

    return metadata


def get_chunk_id(chunk):
    """Build a stable ID from chunk provenance and text."""
    metadata = get_metadata(chunk)

    existing_id = get_field(chunk, "chunk_id")

    if existing_id is None:
        existing_id = get_field(chunk, "id")

    if existing_id is None:
        existing_id = metadata.get("chunk_id")

    if existing_id is not None:
        return str(existing_id)

    source_id = metadata.get("source_id", "")
    page_number = metadata.get("page_number", "")
    chunk_number = metadata.get("chunk_number", "")
    text = get_text(chunk)

    if not source_id or not text:
        raise ValueError(
            "Cannot identify a chunk reliably. "
            f"Type: {type(chunk).__name__}; "
            f"metadata keys: {list(metadata.keys())}; "
            f"preview: {repr(chunk)[:500]}"
        )

    return f"{source_id}|{page_number}|{chunk_number}|{text}"


def load_cases():
    """Load primary cases from the evaluation JSON file."""
    if not CASES_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {CASES_FILE}"
        )

    with CASES_FILE.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError(
            "Evaluation JSON must contain a non-empty list."
        )

    required = {
        "id",
        "question",
        "expected_source_id",
        "expected_pdf_pages",
    }

    for case in cases:
        missing = required - set(case)

        if missing:
            raise ValueError(
                f"Case {case.get('id', '<unknown>')} "
                f"is missing fields: {sorted(missing)}"
            )

    return cases


def match_dense_result(result, chunks, chunks_by_id):
    """Map a dense search result back to its indexed source chunk."""
    result_text = get_text(result)
    result_metadata = get_metadata(result)

    source_id = result_metadata.get("source_id")
    page_number = result_metadata.get("page_number")

    try:
        result_id = get_chunk_id(result)
    except ValueError:
        result_id = None

    if result_id in chunks_by_id:
        return result_id

    matches = [
        get_chunk_id(chunk)
        for chunk in chunks
        if (
            get_metadata(chunk).get("source_id") == source_id
            and get_metadata(chunk).get("page_number") == page_number
            and get_text(chunk) == result_text
        )
    ]

    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        "Could not uniquely map a dense result to an indexed chunk. "
        f"Source={source_id}, page={page_number}, "
        f"matching chunks={len(matches)}. "
        f"Result preview: {repr(result)[:500]}"
    )


def retrieve_dense(question, dense_retriever, chunks, chunks_by_id):
    """Return dense retrieval results as stable chunk IDs."""
    results = dense_retriever.search(question, top_k=TOP_K)

    if not isinstance(results, (list, tuple)):
        raise TypeError(
            "Expected dense search to return a list or tuple; "
            f"received {type(results).__name__}."
        )

    return [
        match_dense_result(result, chunks, chunks_by_id)
        for result in results
    ]


def retrieve_lexical(question, vectorizer, lexical_matrix, chunks):
    """Return TF-IDF retrieval results as stable chunk IDs."""
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(
        query_vector, lexical_matrix
    ).ravel()

    indices = scores.argsort()[::-1][:TOP_K]

    return [
        get_chunk_id(chunks[index])
        for index in indices
    ]


def reciprocal_rank_fusion(dense_ids, lexical_ids):
    """Fuse ranked lists using Reciprocal Rank Fusion."""
    scores = defaultdict(float)

    for rank, chunk_id in enumerate(dense_ids, start=1):
        scores[chunk_id] += 1.0 / (RRF_K + rank)

    for rank, chunk_id in enumerate(lexical_ids, start=1):
        scores[chunk_id] += 1.0 / (RRF_K + rank)

    return sorted(
        scores,
        key=scores.get,
        reverse=True,
    )[:TOP_K]


def get_target_ranks(result_ids, chunks_by_id, source_id, pages):
    """Return ranks containing the expected source and page(s)."""
    target_pages = set(pages)
    ranks = []

    for rank, chunk_id in enumerate(result_ids, start=1):
        metadata = get_metadata(chunks_by_id[chunk_id])

        if (
            metadata.get("source_id") == source_id
            and metadata.get("page_number") in target_pages
        ):
            ranks.append(rank)

    return ranks


def print_case_results(
    case,
    question,
    result_lists,
    chunks_by_id,
    label,
):
    """Print expected-page ranks, Hit@5, MRR, and top-result text."""
    source_id = case["expected_source_id"]
    pages = case["expected_pdf_pages"]

    print("\n" + "=" * 78)
    print(f"{label} | {case['id']}: {question}")
    print(
        f"Expected evidence: {source_id}, "
        f"physical PDF page(s) {pages}"
    )

    for method_name, result_ids in result_lists.items():
        ranks = get_target_ranks(
            result_ids,
            chunks_by_id,
            source_id,
            pages,
        )

        first_rank = ranks[0] if ranks else None
        hit_at_5 = any(rank <= METRIC_K for rank in ranks)
        reciprocal_rank = (
            1.0 / first_rank if first_rank else 0.0
        )

        print(f"\n{method_name}")
        print(
            "Expected-page rank(s):",
            ranks if ranks else f"not in top {TOP_K}",
        )
        print(f"Hit Rate@{METRIC_K}: {int(hit_at_5)}")
        print(
            f"Reciprocal rank (within top {TOP_K}): "
            f"{reciprocal_rank:.3f}"
        )

        for rank, chunk_id in enumerate(result_ids[:5], start=1):
            chunk = chunks_by_id[chunk_id]
            metadata = get_metadata(chunk)
            text = " ".join(get_text(chunk).split())

            print(
                f"  {rank}. {metadata.get('source_id')} | "
                f"Page {metadata.get('page_number')} | "
                f"Chunk {metadata.get('chunk_number')}"
            )

            print(f"     Text: {text[:PREVIEW_LENGTH]}")

            if len(text) > PREVIEW_LENGTH:
                print("     [Preview truncated]")


def evaluate_question(
    case,
    question,
    label,
    dense_retriever,
    vectorizer,
    lexical_matrix,
    chunks,
    chunks_by_id,
):
    """Run all three retrieval methods for one question."""
    dense_ids = retrieve_dense(
        question,
        dense_retriever,
        chunks,
        chunks_by_id,
    )

    lexical_ids = retrieve_lexical(
        question,
        vectorizer,
        lexical_matrix,
        chunks,
    )

    hybrid_ids = reciprocal_rank_fusion(
        dense_ids,
        lexical_ids,
    )

    result_lists = {
        "DENSE": dense_ids,
        "TF-IDF": lexical_ids,
        "HYBRID-RRF": hybrid_ids,
    }

    print_case_results(
        case,
        question,
        result_lists,
        chunks_by_id,
        label,
    )

    results = {}

    for method, ids in result_lists.items():
        ranks = get_target_ranks(
            ids,
            chunks_by_id,
            case["expected_source_id"],
            case["expected_pdf_pages"],
        )

        results[method] = {
            "hit_at_5": int(
                bool(ranks and ranks[0] <= METRIC_K)
            ),
            "reciprocal_rank": (
                1.0 / ranks[0] if ranks else 0.0
            ),
        }

    return results


def print_summary(summary, title):
    """Print mean Hit@5 and MRR across evaluated questions."""
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)

    if not summary:
        print("No cases were evaluated.")
        return

    methods = ["DENSE", "TF-IDF", "HYBRID-RRF"]

    print(f"Number of questions: {len(summary)}")
    print(
        f"Hit Rate@{METRIC_K} and MRR use the expected "
        "source/page annotation."
    )
    print()
    print(f"{'Method':<14} {'Hit Rate@5':>12} {'MRR':>10}")

    for method in methods:
        hit_rate = sum(
            row[method]["hit_at_5"] for row in summary
        ) / len(summary)

        mrr = sum(
            row[method]["reciprocal_rank"] for row in summary
        ) / len(summary)

        print(
            f"{method:<14} "
            f"{hit_rate:>12.3f} "
            f"{mrr:>10.3f}"
        )


def main():
    cases = load_cases()

    print("Loading registered guideline documents...")
    chunks, _ = ingest_registered_guidelines()

    texts = [get_text(chunk) for chunk in chunks]

    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Some indexed chunks have empty text.")

    print(f"Indexed chunks: {len(chunks)}")

    chunks_by_id = {
        get_chunk_id(chunk): chunk
        for chunk in chunks
    }

    print("\nBuilding dense retriever...")
    dense_retriever, _ = build_registered_retriever()

    print("\nBuilding TF-IDF index...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    lexical_matrix = vectorizer.fit_transform(texts)

    cases_by_id = {
        case["id"]: case
        for case in cases
    }

    # ---------------------------------------------------------
    # 1. PRIMARY BENCHMARK: original five questions
    # ---------------------------------------------------------
    primary_summary = []

    print("\n\n### PRIMARY BENCHMARK: ORIGINAL QUESTIONS ###")

    for case in cases:
        question = case.get(
            "original_question",
            case["question"],
        )

        row = evaluate_question(
            case=case,
            question=question,
            label="PRIMARY",
            dense_retriever=dense_retriever,
            vectorizer=vectorizer,
            lexical_matrix=lexical_matrix,
            chunks=chunks,
            chunks_by_id=chunks_by_id,
        )

        primary_summary.append(row)

    print_summary(
        primary_summary,
        "PRIMARY BENCHMARK SUMMARY",
    )

    # ---------------------------------------------------------
    # 2. DIAGNOSTIC: alternate wording
    # ---------------------------------------------------------
    diagnostic_summary = []

    print("\n\n### DIAGNOSTIC EXPERIMENT: ALTERNATE WORDING ###")

    for case_id, diagnostic_question in DIAGNOSTIC_QUESTIONS.items():
        case = cases_by_id.get(case_id)

        if case is None:
            print(
                f"Skipping {case_id}: "
                "case not found in evaluation JSON."
            )
            continue

        row = evaluate_question(
            case=case,
            question=diagnostic_question,
            label="DIAGNOSTIC ONLY",
            dense_retriever=dense_retriever,
            vectorizer=vectorizer,
            lexical_matrix=lexical_matrix,
            chunks=chunks,
            chunks_by_id=chunks_by_id,
        )

        diagnostic_summary.append(row)

    print_summary(
        diagnostic_summary,
        "DIAGNOSTIC VARIANTS SUMMARY (NOT PART OF PRIMARY SCORE)",
    )

    # ---------------------------------------------------------
    # 3. DIAGNOSTIC: manually expanded queries
    # ---------------------------------------------------------
    expanded_summary = []

    print("\n\n### DIAGNOSTIC EXPERIMENT: QUERY EXPANSION ###")

    for case_id, expanded_question in EXPANDED_QUESTIONS.items():
        case = cases_by_id.get(case_id)

        if case is None:
            print(
                f"Skipping {case_id}: "
                "case not found in evaluation JSON."
            )
            continue

        row = evaluate_question(
            case=case,
            question=expanded_question,
            label="QUERY-EXPANSION DIAGNOSTIC ONLY",
            dense_retriever=dense_retriever,
            vectorizer=vectorizer,
            lexical_matrix=lexical_matrix,
            chunks=chunks,
            chunks_by_id=chunks_by_id,
        )

        expanded_summary.append(row)

    print_summary(
        expanded_summary,
        "QUERY-EXPANSION SUMMARY (NOT PART OF PRIMARY SCORE)",
    )

    print("\nExperiment complete.")
    print(
        "Interpretation: these are retrieval metrics against expected "
        "source/page annotations, not measures of clinical answer "
        "correctness or patient safety."
    )
    print(
        "The expanded queries are manually authored diagnostic probes; "
        "they do not test an automated query-expansion component."
    )


if __name__ == "__main__":
    main()