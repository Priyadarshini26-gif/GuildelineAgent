
import sys
from pathlib import Path

# Add the project root so Python can find the src package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_pipeline import build_registered_retriever


TOP_K = 10
RECALL_K = 5
MRR_K = 10


EVALUATION_CASES = [
    {
        "question": (
            "According to NICE NG111, what should be done "
            "with a urine sample for people aged 16 and over "
            "with acute pyelonephritis?"
        ),
        "expected_source": "nice_ng111",
        "relevant_pages": [6],
        "relevance_note": (
            "Physical PDF page 6 recommends obtaining a "
            "midstream urine sample before antibiotics and "
            "sending it for culture and susceptibility testing."
        ),
    },
    {
        "question": (
            "According to IDSA/ESCMID 2010, what tests should "
            "be performed in patients suspected of "
            "pyelonephritis, and how should empirical therapy "
            "be adjusted?"
        ),
        "expected_source": "idsa_escmid_2010_uncomplicated",
        "relevant_pages": [13],
        "relevance_note": (
            "Physical PDF page 13 contains Recommendation 8, "
            "which recommends urine culture and susceptibility "
            "testing for suspected pyelonephritis and tailoring "
            "empirical therapy to the infecting uropathogen."
        ),
    },
    {
        "question": (
            "According to IDSA/ESCMID 2010, what initial "
            "intravenous antimicrobial regimens are listed "
            "for women with pyelonephritis requiring "
            "hospitalization?"
        ),
        "expected_source": "idsa_escmid_2010_uncomplicated",
        "relevant_pages": [4],
        "relevance_note": (
            "Physical PDF page 4 contains Recommendation 13, "
            "which lists initial intravenous antimicrobial "
            "regimens for women with pyelonephritis requiring "
            "hospitalization."
        ),
    },
]


def is_relevant_result(result, expected_source, relevant_pages):
    """Check whether a result matches both the expected source and page."""
    metadata = result.get("metadata", {})

    return (
        metadata.get("source_id") == expected_source
        and metadata.get("page_number") in relevant_pages
    )


def calculate_recall_at_k(results, expected_source, relevant_pages, k):
    """
    Calculate page-level Recall@k.

    This evaluation labels relevant physical PDF pages rather than
    individual evidence passages. It measures the fraction of
    labelled pages represented in the top-k results.
    """
    relevant_pages = set(relevant_pages)

    if not relevant_pages:
        raise ValueError("At least one relevant page must be labelled.")

    top_results = results[:k]

    retrieved_relevant_pages = {
        result.get("metadata", {}).get("page_number")
        for result in top_results
        if (
            result.get("metadata", {}).get("source_id")
            == expected_source
            and result.get("metadata", {}).get("page_number")
            in relevant_pages
        )
    }

    return len(retrieved_relevant_pages) / len(relevant_pages)


def calculate_first_relevant_rank(results, expected_source, relevant_pages):
    """Return the 1-based rank of the first relevant result, or None."""
    for rank, result in enumerate(results, start=1):
        if is_relevant_result(result, expected_source, relevant_pages):
            return rank

    return None


def calculate_mrr_at_k(results, expected_source, relevant_pages, k):
    """Calculate the reciprocal rank of the first relevant result in top k."""
    first_rank = calculate_first_relevant_rank(
        results[:k],
        expected_source,
        relevant_pages,
    )

    return 1.0 / first_rank if first_rank is not None else 0.0


def calculate_source_hit_at_k(results, expected_source, k):
    """Check whether the expected source appears in the top-k results."""
    return any(
        result.get("metadata", {}).get("source_id") == expected_source
        for result in results[:k]
    )


def print_results(results, expected_source, relevant_pages, title):
    """Print ranked results and identify relevant evidence pages."""
    print(f"\n{title}")

    if not results:
        print("No results returned.")
        return

    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        source_id = metadata.get("source_id")
        page_number = metadata.get("page_number")
        score = result.get("score", 0.0)

        relevant = is_relevant_result(
            result,
            expected_source,
            relevant_pages,
        )

        print("\n" + "-" * 60)
        print(f"Rank: {rank}")
        print(f"Source: {source_id}")
        print(f"Physical PDF page: {page_number}")
        print(f"Score: {score:.4f}")
        print(f"Relevant: {relevant}")
        print(result.get("text", ""))


def main():
    print("Building the guideline retrieval index...")

    retriever, report = build_registered_retriever()

    print("\nGuideline ingestion report:")
    for item in report:
        print(item)

    evaluation_results = []

    for case in EVALUATION_CASES:
        question = case["question"]
        expected_source = case["expected_source"]
        relevant_pages = set(case["relevant_pages"])

        if not relevant_pages:
            raise ValueError(
                f"No relevant pages labelled for: {question}"
            )

        print("\n" + "=" * 80)
        print("EVALUATING QUESTION")
        print(f"Question: {question}")
        print(f"Expected source: {expected_source}")
        print(f"Relevant physical PDF pages: {sorted(relevant_pages)}")

        # 1. Global retrieval: search all registered guidelines.
        results = retriever.search(
            question,
            top_k=TOP_K,
        )

        # 2. Source-filtered retrieval: diagnostic only.
        # This is not used to calculate the main global metrics.
        within_source_results = retriever.search(
            question,
            top_k=TOP_K,
            source_id=expected_source,
        )

        within_source_rank = calculate_first_relevant_rank(
            within_source_results,
            expected_source,
            relevant_pages,
        )

        first_relevant_global_rank = calculate_first_relevant_rank(
            results[:MRR_K],
            expected_source,
            relevant_pages,
        )

        # 3. Calculate global retrieval metrics.
        recall_at_5 = calculate_recall_at_k(
            results,
            expected_source,
            relevant_pages,
            RECALL_K,
        )

        reciprocal_rank_at_10 = calculate_mrr_at_k(
            results,
            expected_source,
            relevant_pages,
            MRR_K,
        )

        source_hit_at_5 = calculate_source_hit_at_k(
            results,
            expected_source,
            RECALL_K,
        )

        evaluation_results.append(
            {
                "question": question,
                "recall_at_5": recall_at_5,
                "reciprocal_rank_at_10": reciprocal_rank_at_10,
                "source_hit_at_5": source_hit_at_5,
                "first_relevant_global_rank": first_relevant_global_rank,
                "within_source_rank": within_source_rank,
            }
        )

        # 4. Print the source-filtered diagnostic.
        print("\n" + "=" * 80)
        print("SOURCE-FILTERED RETRIEVAL DIAGNOSTIC")
        print(
            "First relevant rank within expected source:",
            within_source_rank,
        )

        print_results(
            within_source_results,
            expected_source,
            relevant_pages,
            "Top results within expected source:",
        )

        # 5. Print the main global evaluation.
        print("\n" + "=" * 80)
        print("GLOBAL RETRIEVAL EVALUATION")
        print(f"Recall@{RECALL_K}: {recall_at_5:.3f}")
        print(f"Source Hit@{RECALL_K}: {source_hit_at_5}")
        print(
            f"First relevant global rank within top {MRR_K}: "
            f"{first_relevant_global_rank}"
        )
        print(f"Reciprocal rank contribution: {reciprocal_rank_at_10:.3f}")
        print(f"Evaluation note: {case['relevance_note']}")

        print_results(
            results,
            expected_source,
            relevant_pages,
            "Top global retrieval results:",
        )

    # 6. Calculate aggregate metrics.
    if not evaluation_results:
        raise ValueError("No evaluation questions were provided.")

    case_count = len(evaluation_results)

    mean_recall_at_5 = sum(
        item["recall_at_5"]
        for item in evaluation_results
    ) / case_count

    mrr_at_10 = sum(
        item["reciprocal_rank_at_10"]
        for item in evaluation_results
    ) / case_count

    source_hit_rate_at_5 = sum(
        item["source_hit_at_5"]
        for item in evaluation_results
    ) / case_count

    print("\n" + "=" * 80)
    print("FINAL RETRIEVAL EVALUATION")
    print(f"Number of questions: {case_count}")
    print(f"Mean Recall@{RECALL_K}: {mean_recall_at_5:.3f}")
    print(f"MRR@{MRR_K}: {mrr_at_10:.3f}")
    print(
        f"Source Hit Rate@{RECALL_K}: "
        f"{source_hit_rate_at_5:.1%}"
    )

    print("\nDIAGNOSTIC SUMMARY")

    for index, item in enumerate(evaluation_results, start=1):
        print(
            f"Question {index}: "
            f"global rank={item['first_relevant_global_rank']}, "
            f"within-source rank={item['within_source_rank']}"
        )

    print(
        "\nInterpretation: Recall is page-level for this benchmark. "
        "Source Hit@5 checks document retrieval, not relevant evidence "
        "retrieval. These metrics do not validate clinical answers, "
        "guideline currency, or clinical safety."
    )


if __name__ == "__main__":
    main()
