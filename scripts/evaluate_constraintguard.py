
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
    """Find matching evidence and record its one-based retrieval rank."""
    matches = []

    for rank, item in enumerate(evidence, start=1):
        if not isinstance(item, dict):
            continue

        source_id = get_field(item, "source_id")
        page_number = normalize_page(get_field(item, "page_number"))

        if source_id == expected_source and page_number in expected_pages:
            matches.append({
                "rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "source_id": source_id,
                "page_number": page_number,
                "chunk_id": get_field(item, "chunk_id"),
            })

    return matches


def first_match_rank(matches):
    """Return the first matching rank, or None if no match exists."""
    return matches[0]["rank"] if matches else None


def reciprocal_rank(matches):
    """Return reciprocal rank of the first match, or zero for a miss."""
    rank = first_match_rank(matches)
    return 1.0 / rank if rank is not None else 0.0


def summarize_results(results, hit_key):
    """Calculate hit rate, mean reciprocal rank, and rank statistics."""
    total = len(results)
    hits = sum(bool(item[hit_key]) for item in results)

    reciprocal_ranks = [
        item["primary_reciprocal_rank"]
        for item in results
    ]

    found_ranks = [
        item["primary_rank"]
        for item in results
        if item["primary_rank"] is not None
    ]

    return {
        "cases": total,
        "source_and_page_hits": hits,
        "hit_rate": hits / total if total else 0.0,
        "mean_reciprocal_rank": (
            sum(reciprocal_ranks) / total if total else 0.0
        ),
        "mean_rank_when_found": (
            sum(found_ranks) / len(found_ranks) if found_ranks else None
        ),
    }


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

    # Supporting matches never count as primary-page hits.
    supporting_matches = score_evidence(
        evidence=evidence,
        expected_source=expected_source,
        expected_pages=supporting_pages,
    )

    primary_rank = first_match_rank(primary_matches)
    primary_reciprocal_rank = reciprocal_rank(primary_matches)

    supporting_rank = first_match_rank(supporting_matches)
    supporting_reciprocal_rank = reciprocal_rank(supporting_matches)

    retrieved_citations = []

    for rank, item in enumerate(evidence, start=1):
        if not isinstance(item, dict):
            continue

        retrieved_citations.append({
            "rank": rank,
            "source_id": get_field(item, "source_id"),
            "page_number": normalize_page(
                get_field(item, "page_number")
            ),
            "chunk_id": get_field(item, "chunk_id"),
        })

    return {
        "case_id": case["id"],
        "evaluation_type": evaluation_type,
        "question": question,
        "topic": case.get("topic"),
        "expected_source_id": expected_source,
        "expected_pdf_pages": expected_pages,
        "target_source_and_page_found": bool(primary_matches),
        "primary_rank": primary_rank,
        "primary_reciprocal_rank": primary_reciprocal_rank,
        "matching_evidence": primary_matches,
        "expected_supporting_pdf_pages": supporting_pages,
        "supporting_page_evaluation_applicable": bool(supporting_pages),
        "supporting_source_and_page_found": (
            bool(supporting_matches) if supporting_pages else None
        ),
        "supporting_rank": (
            supporting_rank if supporting_pages else None
        ),
        "supporting_reciprocal_rank": (
            supporting_reciprocal_rank if supporting_pages else None
        ),
        "supporting_matching_evidence": supporting_matches,
        "retrieved_evidence_count": len(evidence),
        "retrieved_citations": retrieved_citations,
        "retrieval_status": retrieval.get("status"),
        "expansion_triggered": retrieval.get("expansion_triggered"),
        "expansion_reasons": retrieval.get("expansion_reasons", []),
        "constraint_status": result["constraint_report"].get("status"),
        "answer_status": result["answer"].get("status"),
        "scope_note": case.get("scope_note"),
    }


def summarize_supporting_results(results):
    """Summarize supporting-page performance on applicable cases only."""
    applicable = [
        item for item in results
        if item["supporting_page_evaluation_applicable"]
    ]

    total = len(applicable)
    hits = sum(
        item["supporting_source_and_page_found"]
        for item in applicable
    )

    return {
        "applicable_cases": total,
        "supporting_source_and_page_hits": hits,
        "hit_rate": hits / total if total else None,
        "mean_reciprocal_rank": (
            sum(item["supporting_reciprocal_rank"] for item in applicable)
            / total
            if total else None
        ),
        "mean_rank_when_found": (
            sum(
                item["supporting_rank"]
                for item in applicable
                if item["supporting_rank"] is not None
            )
            / hits
            if hits else None
        ),
    }


def main():
    with CASES_FILE.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    primary_results = []
    diagnostic_results = []

    for case in cases:
        # Preserve original wording in the primary benchmark.
        primary_question = case.get("original_question", case["question"])

        print(f"Evaluating primary case {case['id']}...")
        primary_results.append(
            evaluate_case(case, primary_question, "primary")
        )

        # Alternative wording is evaluated separately.
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

    primary_summary = summarize_results(
        primary_results,
        "target_source_and_page_found",
    )

    supporting_summary = summarize_supporting_results(primary_results)

    diagnostic_summary = summarize_results(
        diagnostic_results,
        "target_source_and_page_found",
    )

    report = {
        "project": "ConstraintGuard-ED",
        "evaluation_note": (
            "Retrieval matching only. A source-and-page match does not "
            "establish clinical correctness, population applicability, "
            "or safety. Rank is the one-based position in the retrieved "
            "evidence list. Reciprocal rank is 1/rank for the first match "
            "and zero when the target is absent. Supporting-page hits are "
            "reported separately and do not count as primary-page hits."
        ),
        "primary_summary": primary_summary,
        "supporting_summary": supporting_summary,
        "primary_results": primary_results,
        "diagnostic_summary": diagnostic_summary,
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
        rank_text = (
            str(item["primary_rank"])
            if item["primary_rank"] is not None
            else "not found"
        )

        print(
            f"{item['case_id']}: {status} | "
            f"target={item['expected_source_id']} "
            f"page(s)={item['expected_pdf_pages']} | "
            f"rank={rank_text} | "
            f"RR={item['primary_reciprocal_rank']:.3f} | "
            f"evidence={item['retrieved_evidence_count']}"
        )

        if item["supporting_page_evaluation_applicable"]:
            supporting_status = (
                "HIT"
                if item["supporting_source_and_page_found"]
                else "MISS"
            )
            supporting_rank_text = (
                str(item["supporting_rank"])
                if item["supporting_rank"] is not None
                else "not found"
            )

            print(
                f"  Supporting pages "
                f"{item['expected_supporting_pdf_pages']}: "
                f"{supporting_status} | "
                f"rank={supporting_rank_text} | "
                f"RR={item['supporting_reciprocal_rank']:.3f}"
            )

    print(
        f"\nPrimary source-and-page hit rate: "
        f"{primary_summary['source_and_page_hits']}/"
        f"{primary_summary['cases']} "
        f"({primary_summary['hit_rate']:.1%})"
    )
    print(
        "Primary mean reciprocal rank (MRR): "
        f"{primary_summary['mean_reciprocal_rank']:.3f}"
    )
    print(
        "Mean primary rank when found: "
        f"{primary_summary['mean_rank_when_found']}"
    )

    if supporting_summary["applicable_cases"]:
        print(
            f"\nSupporting source-and-page hit rate: "
            f"{supporting_summary['supporting_source_and_page_hits']}/"
            f"{supporting_summary['applicable_cases']} "
            f"({supporting_summary['hit_rate']:.1%})"
        )
        print(
            "Supporting mean reciprocal rank (MRR): "
            f"{supporting_summary['mean_reciprocal_rank']:.3f}"
        )
    else:
        print("\nSupporting source-and-page hit rate: N/A")

    print(
        f"\nDiagnostic variants evaluated: "
        f"{diagnostic_summary['cases']}"
    )
    print(
        f"Diagnostic source-and-page hits: "
        f"{diagnostic_summary['source_and_page_hits']}/"
        f"{diagnostic_summary['cases']} "
        f"({diagnostic_summary['hit_rate']:.1%})"
    )
    print(
        "Diagnostic mean reciprocal rank (MRR): "
        f"{diagnostic_summary['mean_reciprocal_rank']:.3f}"
    )
    print(f"Detailed report saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
