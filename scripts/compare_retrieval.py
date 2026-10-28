
"""
Pilot comparison of baseline, rewritten, and trigger-based retrieval.

This is an exploratory retrieval experiment, not a clinical safety evaluation.
"""

import json
from pathlib import Path

from src.retrieval_pipeline import build_registered_retriever
from src.risk_triggered_retrieval import retrieve_with_risk_trigger


CASES_FILE = Path("data/evaluation/constraintguard_cases.json")

COMPLETE_CONTEXT = {
    "age": 45,
    "allergies": "no",
    "pregnancy_status": "not applicable",
    "renal_function": "normal",
    "severity": "stable",
    "oral_tolerance": "yes",
    "culture_results": "pending",
}

TOP_K = 8
INITIAL_TOP_K = 3
EXPANDED_TOP_K = 8


def load_cases():
    """Load and validate evaluation cases from the JSON dataset."""
    if not CASES_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {CASES_FILE}\n"
            "Run this script from the inner GuildelineAgent project directory."
        )

    with CASES_FILE.open("r", encoding="utf-8") as file:
        raw_cases = json.load(file)

    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("The evaluation dataset must be a non-empty JSON list.")

    cases = []
    seen_ids = set()

    for item in raw_cases:
        required_fields = {
            "id",
            "question",
            "expected_source_id",
            "expected_pdf_pages",
        }
        missing = required_fields - item.keys()

        if missing:
            raise ValueError(
                f"Case is missing required fields {sorted(missing)}: {item}"
            )

        case_id = item["id"]
        question = item["question"]
        source_id = item["expected_source_id"]
        pages = item["expected_pdf_pages"]

        if case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case ID: {case_id}")
        seen_ids.add(case_id)

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{case_id}: question must be a non-empty string.")

        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"{case_id}: expected_source_id must be a string.")

        if not isinstance(pages, list) or not pages:
            raise ValueError(
                f"{case_id}: expected_pdf_pages must be a non-empty list."
            )

        try:
            pages = [int(page) for page in pages]
        except (TypeError, ValueError):
            raise ValueError(
                f"{case_id}: expected_pdf_pages must contain page numbers."
            ) from None

        if any(page < 1 for page in pages):
            raise ValueError(f"{case_id}: PDF page numbers must be positive.")

        # Preserve optional manually written query rewrites, if supplied.
        rewritten_question = item.get("rewritten_question")

        if rewritten_question is not None:
            if (
                not isinstance(rewritten_question, str)
                or not rewritten_question.strip()
            ):
                raise ValueError(
                    f"{case_id}: rewritten_question must be non-empty "
                    "when provided."
                )

        cases.append({
            "id": case_id,
            "name": f"{case_id} - {item.get('topic', 'evaluation case')}",
            "question": question.strip(),
            "rewritten_question": rewritten_question,
            "source_id": source_id,
            "pages": pages,
            "topic": item.get("topic", "unspecified"),
            "scope_note": item.get("scope_note"),
        })

    return cases


def get_source_page(item):
    """Return source ID and page number from a retrieved item."""
    metadata = item.get("metadata", {})

    source_id = item.get("source_id") or metadata.get("source_id")
    page = item.get("page_number") or metadata.get("page_number")

    return source_id, page


def is_relevant(item, case):
    """Check whether an item matches any expected page for its source."""
    source_id, page = get_source_page(item)

    try:
        page = int(page)
    except (TypeError, ValueError):
        return False

    return source_id == case["source_id"] and page in case["pages"]


def relevant_rank(results, case):
    """Return the 1-based rank of the first expected page, or None."""
    for rank, item in enumerate(results, start=1):
        if is_relevant(item, case):
            return rank

    return None


def recall_at_5(results, case):
    """
    Binary success at 5: whether any expected page appears in the top five.

    This is a hit-rate measure, not conventional multi-document recall.
    """
    return float(any(is_relevant(item, case) for item in results[:5]))


def reciprocal_rank_at_8(results, case):
    """Reciprocal rank of the first expected page, evaluated through rank 8."""
    rank = relevant_rank(results[:TOP_K], case)
    return 1.0 / rank if rank is not None else 0.0


def source_hit_at_5(results, case):
    """Check source-document presence, irrespective of exact page."""
    return float(
        any(
            get_source_page(item)[0] == case["source_id"]
            for item in results[:5]
        )
    )


def summarize(name, rows):
    """Print aggregate retrieval metrics for a system."""
    if not rows:
        print(f"\n{name}: no results to evaluate.")
        return

    count = len(rows)

    page_hit_rate = sum(
        recall_at_5(results, case)
        for results, case in rows
    ) / count

    mrr = sum(
        reciprocal_rank_at_8(results, case)
        for results, case in rows
    ) / count

    source_hit_rate = sum(
        source_hit_at_5(results, case)
        for results, case in rows
    ) / count

    print(f"\n{name}")
    print(f"  Expected-page Hit Rate@5: {page_hit_rate:.3f}")
    print(f"  MRR@8 (expected page): {mrr:.3f}")
    print(f"  Source Hit Rate@5: {source_hit_rate:.1%}")


def retrieve_expanded_results(retriever, question, trigger_result, baseline):
    """
    Score the retriever's top-8 ranking when expansion is triggered.

    If expansion is not triggered, preserve the initial top-3 result list.
    This evaluates ranking after a trigger, not the exact ordering of the
    trigger function's combined result list.
    """
    if trigger_result.get("expansion_triggered"):
        return retriever.search(question, top_k=EXPANDED_TOP_K)

    return baseline[:INITIAL_TOP_K]


def evaluate():
    """Run the pilot comparison using the JSON evaluation dataset."""
    cases = load_cases()
    retriever, report = build_registered_retriever()

    systems = {
        "Baseline (original query, top 8)": [],
        "Risk-triggered retrieval (up to top 8)": [],
        "Combined triggers (up to top 8)": [],
    }

    rewritten_cases = [
        case for case in cases
        if case["rewritten_question"] is not None
    ]

    if rewritten_cases:
        systems["Manual query rewriting (top 8)"] = []

    print("INDEX:", report)
    print(f"DATASET: {CASES_FILE}")
    print(f"NUMBER OF EVALUATION CASES: {len(cases)}")
    print("\nPER-QUESTION RESULTS")

    for case in cases:
        question = case["question"]

        # Baseline: original evaluation question.
        baseline = retriever.search(question, top_k=TOP_K)
        systems["Baseline (original query, top 8)"].append(
            (baseline, case)
        )

        # Rewritten-query condition is run only when a rewrite is supplied.
        if case["rewritten_question"] is not None:
            rewritten = retriever.search(
                case["rewritten_question"],
                top_k=TOP_K,
            )
            systems["Manual query rewriting (top 8)"].append(
                (rewritten, case)
            )

        # Risk-triggered retrieval with complete example context.
        risk_result = retrieve_with_risk_trigger(
            retriever,
            question,
            patient_context=COMPLETE_CONTEXT,
            initial_top_k=INITIAL_TOP_K,
            expanded_top_k=EXPANDED_TOP_K,
        )

        risk_results = retrieve_expanded_results(
            retriever,
            question,
            risk_result,
            baseline,
        )

        systems["Risk-triggered retrieval (up to top 8)"].append(
            (risk_results, case)
        )

        # Combined condition: original query and missing patient context.
        combined_result = retrieve_with_risk_trigger(
            retriever,
            question,
            patient_context={},
            initial_top_k=INITIAL_TOP_K,
            expanded_top_k=EXPANDED_TOP_K,
        )

        combined_results = retrieve_expanded_results(
            retriever,
            question,
            combined_result,
            baseline,
        )

        systems["Combined triggers (up to top 8)"].append(
            (combined_results, case)
        )

        print(f"\n{case['name']}")
        print(f"  Expected source: {case['source_id']}")
        print(f"  Expected PDF page(s): {case['pages']}")

        if case["scope_note"]:
            print(f"  Scope note: {case['scope_note']}")

        print(
            "  Baseline expected-page rank:",
            relevant_rank(baseline, case),
        )

        if case["rewritten_question"] is not None:
            print(
                "  Rewritten-query expected-page rank:",
                relevant_rank(rewritten, case),
            )

        print(
            "  Risk-triggered expected-page rank:",
            relevant_rank(risk_results, case),
        )
        print(
            "  Combined expected-page rank:",
            relevant_rank(combined_results, case),
        )
        print(
            "  Risk-only triggers:",
            risk_result.get("expansion_reasons"),
        )
        print(
            "  Combined triggers:",
            combined_result.get("expansion_reasons"),
        )

    print("\nAGGREGATE METRICS")
    for name, rows in systems.items():
        summarize(name, rows)

    if not rewritten_cases:
        print(
            "\nMANUAL QUERY REWRITING: NOT EVALUATED\n"
            "No cases in the JSON dataset contain a rewritten_question field. "
            "Add independently authored rewrites before comparing this system."
        )

    print(
        "\nLIMITATIONS:\n"
        f"- This is an exploratory pilot with {len(cases)} questions.\n"
        "- Expected source IDs and PDF pages are manually labelled; "
        "incorrect labels affect the metrics.\n"
        "- Expected-page Hit Rate@5 measures whether at least one labelled "
        "page appears in the top five; it is not conventional recall over "
        "a complete set of relevant documents.\n"
        "- MRR@8 evaluates the first expected page found in the top eight.\n"
        "- Source Hit Rate@5 checks source-document presence, not exact-page "
        "relevance.\n"
        "- Baseline results contain up to eight items. Triggered conditions "
        "contain three items when expansion does not occur and up to eight "
        "when it does.\n"
        "- When expansion occurs, scoring uses a direct top-8 search rather "
        "than the trigger function's combined result ordering.\n"
        "- The complete patient context is a synthetic example, not a real "
        "patient record or a validated clinical scenario.\n"
        "- Missing-context flags do not establish clinical suitability "
        "or safety.\n"
        "- Retrieval metrics do not measure answer correctness or patient "
        "safety.\n"
        "- This script does not establish clinical validation or superiority "
        "of any retrieval approach."
    )


if __name__ == "__main__":
    evaluate()
