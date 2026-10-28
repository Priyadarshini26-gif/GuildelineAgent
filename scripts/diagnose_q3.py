
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_pipeline import build_registered_retriever

SOURCE = "idsa_escmid_2010_uncomplicated"
PAGE = 4
OUT = PROJECT_ROOT / "diagnostic_q3_phrasings.txt"

QUESTIONS = {
    "original": (
        "According to IDSA/ESCMID 2010, what initial intravenous "
        "antimicrobial regimens are listed for women with "
        "pyelonephritis requiring hospitalization?"
    ),
    "guideline wording": (
        "What intravenous antimicrobial regimen should be used initially "
        "for women hospitalized with pyelonephritis?"
    ),
    "recommendation wording": (
        "Which initial IV antimicrobial options are recommended for "
        "hospitalized women with pyelonephritis in the IDSA/ESCMID guideline?"
    ),
    "specific regimen terms": (
        "For hospitalized women with pyelonephritis, which initial "
        "fluoroquinolone, aminoglycoside, cephalosporin, penicillin, "
        "or carbapenem regimens are discussed?"
    ),
}


def relevant_rank(results):
    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        if (
            metadata["source_id"] == SOURCE
            and metadata["page_number"] == PAGE
        ):
            return rank
    return None


def main():
    retriever, _ = build_registered_retriever()
    lines = []

    for label, question in QUESTIONS.items():
        # Search the entire index and the expected source separately.
        global_results = retriever.search(question, top_k=20)
        source_results = retriever.search(
            question, top_k=20, source_id=SOURCE
        )

        global_rank = relevant_rank(global_results)
        source_rank = relevant_rank(source_results)

        lines.append("=" * 78)
        lines.append(f"TEST: {label}")
        lines.append(f"QUESTION: {question}")
        lines.append(f"Relevant page rank in global top 20: {global_rank}")
        lines.append(f"Relevant page rank within source top 20: {source_rank}")

        lines.append("\nGLOBAL TOP 10:")
        for rank, result in enumerate(global_results[:10], start=1):
            metadata = result["metadata"]
            marker = (
                " <-- RELEVANT PAGE"
                if metadata["source_id"] == SOURCE
                and metadata["page_number"] == PAGE
                else ""
            )
            lines.append(
                f"{rank}. source={metadata['source_id']} "
                f"page={metadata['page_number']} "
                f"chunk={metadata['chunk_number']} "
                f"score={result['score']:.4f}{marker}"
            )
            lines.append(result["text"][:220].replace("\n", " "))

        lines.append("\nTOP 10 WITHIN EXPECTED SOURCE:")
        for rank, result in enumerate(source_results[:10], start=1):
            metadata = result["metadata"]
            marker = (
                " <-- RELEVANT PAGE"
                if metadata["page_number"] == PAGE
                else ""
            )
            lines.append(
                f"{rank}. page={metadata['page_number']} "
                f"chunk={metadata['chunk_number']} "
                f"score={result['score']:.4f}{marker}"
            )
            lines.append(result["text"][:220].replace("\n", " "))

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
