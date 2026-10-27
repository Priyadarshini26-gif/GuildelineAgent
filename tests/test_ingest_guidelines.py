
import hashlib
import json

import pymupdf

from src.ingest_guidelines import ingest_registered_guidelines


def create_test_pdf(pdf_path):
    """Create a tiny PDF for testing ingestion."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Antimicrobial guideline test text. "
        "Use evidence to support clinical decisions."
    )
    document.save(pdf_path)
    document.close()


def create_test_record(pdf_path):
    """Create complete metadata for a test guideline."""
    sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    return {
        "id": "test_guideline",
        "title": "Test Antimicrobial Guideline",
        "issuing_organization": "Test Organization",
        "source_url": "https://example.org/test-guideline",
        "publication_date": "2026-01",
        "accessed_date": "2026-10-24",
        "version_notes": "Synthetic test document",
        "scope": "Testing PDF ingestion",
        "exclusions": "Not for clinical use",
        "license_notes": "Synthetic test content",
        "local_filename": pdf_path.name,
        "sha256": sha256,
        "limitations": "Test only; not clinical guidance",
    }


def prepare_test_source(tmp_path):
    guidelines_dir = tmp_path / "guidelines"
    guidelines_dir.mkdir()

    pdf_path = guidelines_dir / "test_guideline.pdf"
    create_test_pdf(pdf_path)

    record = create_test_record(pdf_path)
    registry_path = guidelines_dir / "source_registry.json"
    registry_path.write_text(
        json.dumps([record], indent=2),
        encoding="utf-8",
    )

    return registry_path, guidelines_dir, record


def test_valid_source_is_ingested(tmp_path):
    registry_path, guidelines_dir, record = prepare_test_source(tmp_path)

    chunks, report = ingest_registered_guidelines(
        registry_path=registry_path,
        guidelines_dir=guidelines_dir,
    )

    assert len(chunks) > 0
    assert chunks[0]["metadata"]["source_id"] == record["id"]
    assert report[0]["status"] == "processed"


def test_hash_mismatch_source_is_skipped(tmp_path):
    registry_path, guidelines_dir, record = prepare_test_source(tmp_path)

    record["sha256"] = "0" * 64
    registry_path.write_text(
        json.dumps([record], indent=2),
        encoding="utf-8",
    )

    chunks, report = ingest_registered_guidelines(
        registry_path=registry_path,
        guidelines_dir=guidelines_dir,
    )

    assert chunks == []
    assert report[0]["status"] == "skipped"


def test_unverified_source_is_skipped(tmp_path):
    registry_path, guidelines_dir, record = prepare_test_source(tmp_path)

    record["scope"] = "TO_VERIFY"
    registry_path.write_text(
        json.dumps([record], indent=2),
        encoding="utf-8",
    )

    chunks, report = ingest_registered_guidelines(
        registry_path=registry_path,
        guidelines_dir=guidelines_dir,
    )

    assert chunks == []
    assert report[0]["status"] == "skipped"


def test_missing_pdf_is_skipped(tmp_path):
    registry_path, guidelines_dir, record = prepare_test_source(tmp_path)

    (guidelines_dir / record["local_filename"]).unlink()

    chunks, report = ingest_registered_guidelines(
        registry_path=registry_path,
        guidelines_dir=guidelines_dir,
    )

    assert chunks == []
    assert report[0]["status"] == "skipped"
