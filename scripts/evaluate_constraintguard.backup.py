
"""Batch evaluation for the ConstraintGuard-ED research prototype."""

import json
from pathlib import Path

from src.constraintguard_pipeline import run_constraintguard


ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = ROOT / "data" / "evaluation" / "constraintguard_cases.json"
OUTPUT_FILE = (
    ROOT
    / "data"
    / "evaluation"
    / "constraintguard_evaluation_results.json"
)


def get_field(item, field):
    """Read citation fields from either the item or its metadata."""
    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    value = item.get(field)
    if value is None:
        value = metadata.get(field)

    return value


def normalize_page(page):
    """Convert a page number to an integer when possible."""
    try:
        return int(page) if page is not None else None
    except (TypeError, ValueError):
        return None


def score_evidence(evidence, expected_source, expected_pages):
    """Find evidence matching the expected source and one of its pages."""
    matches = []

    for item in evidence:
        if not isinstance(item, dict):
            continue

        source_id = get_field(item, "source_id")
        page_number = normalize_page(get_field(item, "page_number"))

        if source_id == expected_source and page_number in expected_pages:
            matches.append({
                "source_id": source_id,
                "page_number": page_number,
                "chunk_id": get_field(item, "chunk_id"),
            })

    return matches


def evaluate_case(case, question, evaluation_type):
    """Evaluate primary and supporting evidence separately."""
    result = run_constraintguard(question=question)

    retrieval = result["retrieval"]
    evidence = retrieval.get("evidence", [])

    expected_source = case["expected_source_id"]
    expected_pages = case["expected_pdf_pages"]
    supporting_pages = case.get("expected_supporting_pdf_pages", [])

    # Primary matches determine the primary-page hit metric.
    primary_matches = score_evidence(
        evidence=evidence,
        expected_source=expected_source,
        expected_pages=expected_pages,
    )

    # Supporting matches are reported separately and never count as
    # primary-page hits.
    supporting_matches = score_evidence(
        evidence=evidence,
        expected_source=expected_source,
        expected_pages=supporting_pages,
    )

    return {
        "case_id": case["id"],
        "evaluation_type": evaluation_type,
        "question": question,
        "topic": case.get("topic"),
        "expected_source_id": expected_source,
        "expected_pdf_pages": expected_pages,
        "target_source_and_page_found": bool(primary_matches),
        "matching_evidence": primary_matches,
        "expected_supporting_pdf_pages": supporting_pages,
        "supporting_page_evaluation_applicable": bool(supporting_pages),
        "supporting_source_and_page_found": (
            bool(supporting_matches) if supporting_pages else None
        ),
        "supporting_matching_evidence": supporting_matches,
        "retrieved_evidence_count": len(evidence),
        "retrieved_citations": [
            {
                "source_id": get_field(item, "source_id"),
                "page_number": normalize_page(
                    get_field(item, "page_number")
                ),
                "chunk_id": get_field(item, "chunk_id"),
            }
            for item in evidence
            if isinstance(item, dict)
        ],
        "retrieval_status": retrieval.get("status"),
        "expansion_triggered": retrieval.get("expansion_triggered"),
        "expansion_reasons": retrieval.get("expansion_reasons", []),
        "constraint_status": result["constraint_report"].get("status"),
        "answer_status": result["answer"].get("status"),
        "scope_note": case.get("scope_note"),
    }


def main():
    with CASES_FILE.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    primary_results = []
    diagnostic_results = []

    for case in cases:
        # Keep the original primary-benchmark wording where provided.
        primary_question = case.get("original_question", case["question"])

        print(f"Evaluating primary case {case['id']}...")
        primary_results.append(
            evaluate_case(case, primary_question, "primary")
        )

        # Alternative wording remains separate from primary metrics.
        diagnostic_question = case.get("diagnostic_question")

        if diagnostic_question:
            print(f"Evaluating diagnostic wording for {case['id']}...")
            diagnostic_results.append(
                evaluate_case(
                    case,
                    diagnostic_question,
                    "diagnostic_variant",
                )
            )

        if case.get("evaluation_role") == "diagnostic_variant":
            print(f"Evaluating population-specific wording for {case['id']}...")
            diagnostic_results.append(
                evaluate_case(
                    case,
                    case["question"],
                    "diagnostic_variant",
                )
            )

    primary_total = len(primary_results)
    primary_hits = sum(
        item["target_source_and_page_found"]
        for item in primary_results
    )

    # Only cases that explicitly specify supporting pages are included
    # in the supporting-page metric.
    supporting_applicable = [
        item for item in primary_results
        if item["supporting_page_evaluation_applicable"]
    ]
    supporting_hits = sum(
        item["supporting_source_and_page_found"]
        for item in supporting_applicable
    )
    supporting_total = len(supporting_applicable)

    diagnostic_total = len(diagnostic_results)
    diagnostic_hits = sum(
        item["target_source_and_page_found"]
        for item in diagnostic_results
    )

    report = {
        "project": "ConstraintGuard-ED",
        "evaluation_note": (
            "Retrieval matching only. A source-and-page match does not "
            "establish clinical correctness, population applicability, "
            "or safety. Supporting-page hits are reported separately "
            "and do not count as primary-page hits."
        ),
        "primary_summary": {
            "cases": primary_total,
            "source_and_page_hits": primary_hits,
            "hit_rate": (
                primary_hits / primary_total if primary_total else 0
            ),
        },
        "supporting_summary": {
            "applicable_cases": supporting_total,
            "supporting_source_and_page_hits": supporting_hits,
            "hit_rate": (
                supporting_hits / supporting_total
                if supporting_total else None
            ),
        },
        "primary_results": primary_results,
        "diagnostic_summary": {
            "cases": diagnostic_total,
            "source_and_page_hits": diagnostic_hits,
            "hit_rate": (
                diagnostic_hits / diagnostic_total
                if diagnostic_total else 0
            ),
        },
        "diagnostic_results": diagnostic_results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("CONSTRAINTGUARD-ED EVALUATION")
    print("=" * 60)

    for item in primary_results:
        status = (
            "HIT" if item["target_source_and_page_found"] else "MISS"
        )

        print(
            f"{item['case_id']}: {status} | "
            f"target={item['expected_source_id']} "
            f"page(s)={item['expected_pdf_pages']} | "
            f"evidence={item['retrieved_evidence_count']}"
        )

        if item["supporting_page_evaluation_applicable"]:
            supporting_status = (
                "HIT"
                if item["supporting_source_and_page_found"]
                else "MISS"
            )
            print(
                f"  Supporting pages "
                f"{item['expected_supporting_pdf_pages']}: "
                f"{supporting_status}"
            )

    primary_rate = (
        primary_hits / primary_total if primary_total else 0
    )

    print(
        f"\nPrimary source-and-page hit rate: "
        f"{primary_hits}/{primary_total} ({primary_rate:.1%})"
    )

    if supporting_total:
        print(
            f"Supporting source-and-page hit rate: "
            f"{supporting_hits}/{supporting_total} "
            f"({supporting_hits / supporting_total:.1%})"
        )
    else:
        print("Supporting source-and-page hit rate: N/A")

    print(f"Diagnostic variants evaluated: {diagnostic_total}")
    print(
        f"Diagnostic source-and-page hits: "
        f"{diagnostic_hits}/{diagnostic_total}"
    )
    print(f"Detailed report saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
