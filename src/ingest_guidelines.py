
"""Load verified guideline records and convert eligible PDFs into chunks."""

from pathlib import Path

from src.document_processing import process_pdf
from src.source_registry import (
    GUIDELINES_DIR,
    REGISTRY_PATH,
    check_source,
    load_registry,
)


def ingest_registered_guidelines(
    registry_path: Path = REGISTRY_PATH,
    guidelines_dir: Path = GUIDELINES_DIR,
) -> tuple[list[dict], list[dict]]:
    """Return (chunks, report) for registered guideline documents.

    Records with structural/hash errors or unresolved registry fields
    are skipped. The checks do not establish clinical currency or validity.
    """
    records = load_registry(registry_path)
    all_chunks = []
    report = []

    for record in records:
        source_id = record.get("id", "unknown")
        result = check_source(record, guidelines_dir)

        if result["errors"] or result["unverified"]:
            report.append({
                "source_id": source_id,
                "status": "skipped",
                "errors": result["errors"],
                "unverified": result["unverified"],
                "chunk_count": 0,
            })
            continue

        pdf_path = Path(guidelines_dir) / record["local_filename"]

        try:
            chunks = process_pdf(
                pdf_path=pdf_path,
                source_id=source_id,
                source_title=record["title"],
            )
        except Exception as exc:
            report.append({
                "source_id": source_id,
                "status": "error",
                "errors": [str(exc)],
                "unverified": [],
                "chunk_count": 0,
            })
            continue

        all_chunks.extend(chunks)
        report.append({
            "source_id": source_id,
            "status": "processed" if chunks else "no_text",
            "errors": [],
            "unverified": [],
            "chunk_count": len(chunks),
        })

    return all_chunks, report


def main() -> int:
    chunks, report = ingest_registered_guidelines()

    print("Guideline ingestion report")
    print("=" * 32)

    for item in report:
        print(
            f"{item['source_id']}: {item['status']} "
            f"({item['chunk_count']} chunks)"
        )

        for error in item["errors"]:
            print(f"  ERROR: {error}")

        for field in item["unverified"]:
            print(f"  UNVERIFIED: {field}")

    print("-" * 32)
    print(f"Sources in registry: {len(report)}")
    print(f"Total chunks in memory: {len(chunks)}")

    # Do not silently treat failed ingestion as success.
    failed = any(
        item["status"] in {"skipped", "error", "no_text"}
        for item in report
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
