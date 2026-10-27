import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest_guidelines import ingest_registered_guidelines
from src.retrieval_pipeline import build_registered_retriever

SOURCE = "idsa_escmid_2010_uncomplicated"
PAGE = 4
QUESTION = (
    "According to IDSA/ESCMID 2010, what initial intravenous "
    "antimicrobial regimens are listed for women with "
    "pyelonephritis requiring hospitalization?"
)
KEYWORDS = ["carbapenem", "aminoglycoside", "fluoroquinolone",
            "hospitalization", "hospitalisation", "pyelonephritis"]

OUT = PROJECT_ROOT / "diagnostic_q3.txt"


def main():
    lines = []

    # A. Every chunk the pipeline created for the relevant page.
    chunks, _ = ingest_registered_guidelines()
    page_chunks = [
        c for c in chunks
        if c["metadata"]["source_id"] == SOURCE
        and c["metadata"]["page_number"] == PAGE
    ]
    lines.append(f"=== A. CHUNKS ON {SOURCE} PAGE {PAGE}: {len(page_chunks)} ===")
    for c in page_chunks:
        text = c["text"]
        found = [k for k in KEYWORDS if k in text.lower()]
        lines.append("-" * 70)
        lines.append(f"chunk_id={c['chunk_id']} length={len(text)}")
        lines.append(f"keywords present: {found}")
        lines.append(text)

    # B. Top 10 within the source, with page and chunk number.
    retriever, _ = build_registered_retriever()
    results = retriever.search(QUESTION, top_k=10, source_id=SOURCE)
    lines.append("")
    lines.append("=== B. TOP 10 WITHIN SOURCE ===")
    for rank, r in enumerate(results, start=1):
        m = r["metadata"]
        mark = "  <-- RELEVANT PAGE" if m["page_number"] == PAGE else ""
        lines.append("-" * 70)
        lines.append(
            f"rank={rank} page={m['page_number']} "
            f"chunk={m['chunk_number']} score={r['score']:.4f}{mark}"
        )
        lines.append(r["text"][:400])

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()