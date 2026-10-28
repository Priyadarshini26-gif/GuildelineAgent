
from pathlib import Path

import pymupdf
import pytest

from src.document_processing import (
    chunk_text,
    clean_extracted_text,
    extract_pdf_pages,
    process_pdf,
)


def create_test_pdf(path: Path) -> None:
    """Create a tiny PDF for testing without using real guidelines."""
    with pymupdf.open() as document:
        page1 = document.new_page()
        page1.insert_text((72, 72), "First page contains test text.")

        page2 = document.new_page()
        page2.insert_text((72, 72), "Second page contains more test text.")

        document.save(path)


def test_extract_pdf_pages_preserves_page_numbers(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_test_pdf(pdf_path)

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2
    assert "First page" in pages[0]["text"]
    assert "Second page" in pages[1]["text"]


def test_missing_pdf_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_pdf_pages(tmp_path / "missing.pdf")


def test_chunk_text_returns_empty_for_blank_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_rejects_invalid_settings():
    with pytest.raises(ValueError):
        chunk_text("Some text", chunk_size=0)

    with pytest.raises(ValueError):
        chunk_text("Some text", chunk_size=10, overlap=10)


def test_chunk_text_covers_long_text():
    text = "word " * 500

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_process_pdf_preserves_source_metadata(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_test_pdf(pdf_path)

    chunks = process_pdf(
        pdf_path,
        source_id="test-guideline",
        source_title="Test Guideline",
    )

    assert len(chunks) >= 2

    first = chunks[0]
    assert first["chunk_id"].startswith("test-guideline-p0001")
    assert first["metadata"]["source_id"] == "test-guideline"
    assert first["metadata"]["source_title"] == "Test Guideline"
    assert first["metadata"]["filename"] == "sample.pdf"
    assert first["metadata"]["page_number"] == 1
    assert first["text"]


def test_clean_extracted_text_normalizes_whitespace():
    text = "First\u00a0line\r\nSecond\u200b line\n\n\nThird line "

    cleaned = clean_extracted_text(text)

    assert cleaned == "First line\nSecond line\n\nThird line"

def test_chunk_text_preserves_population_heading_with_recommendation():
    text = (
        "Table 2 Antibiotics for pregnant women aged 12 years and over. "
        "Review intravenous antibiotics by 48 hours and consider "
        "stepping down to oral antibiotics where possible."
    )

    chunks = chunk_text(text, chunk_size=200, overlap=30)

    assert len(chunks) == 1
    assert "pregnant women aged 12 years and over" in chunks[0]
    assert "Review intravenous antibiotics by 48 hours" in chunks[0]
