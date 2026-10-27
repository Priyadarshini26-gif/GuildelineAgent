
"""PDF text extraction and chunking with source and page metadata."""

from pathlib import Path
import re

import pymupdf


DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


def clean_extracted_text(text: str) -> str:
    """Normalize common whitespace and PDF ligature issues."""
    # Normalize non-breaking spaces and invisible characters.
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # Convert common typographic ligatures to ordinary letters.
    text = (
        text.replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\ufb00", "ff")
        .replace("\ufb03", "ffi")
        .replace("\ufb04", "ffl")
    )

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing spaces from each line.
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    text = "\n".join(lines)

    # Reduce excessive blank lines while preserving paragraph breaks.
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)

    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract PDF text while preserving one-based page numbers."""
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = clean_extracted_text(page.get_text("text"))

            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text,
                })

    return pages


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, preferring sentence boundaries."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size"
        )

    # Collapse repeated spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text.strip())

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            # Prefer a sentence boundary in the latter half of the chunk.
            minimum_boundary = start + chunk_size // 2

            sentence_matches = list(
                re.finditer(r"[.!?](?:\s+|$)", text[start:end])
            )

            valid_sentences = [
                match for match in sentence_matches
                if start + match.end() >= minimum_boundary
            ]

            if valid_sentences:
                end = start + valid_sentences[-1].end()
            else:
                # If needed, use a word boundary to avoid splitting a word.
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Keep overlap while ensuring forward progress.
        next_start = max(start + 1, end - overlap)

        # Start at a word boundary where possible.
        whitespace = text.find(" ", next_start, end)
        if whitespace != -1:
            next_start = whitespace + 1

        start = next_start

    return chunks


def process_pdf(
    pdf_path: Path,
    source_id: str,
    source_title: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Extract and chunk a PDF, preserving provenance."""
    pdf_path = Path(pdf_path)
    pages = extract_pdf_pages(pdf_path)
    chunks = []

    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_number, chunk in enumerate(page_chunks, start=1):
            chunks.append({
                "chunk_id": (
                    f"{source_id}-p{page['page_number']:04d}"
                    f"-c{chunk_number:03d}"
                ),
                "text": chunk,
                "metadata": {
                    "source_id": source_id,
                    "source_title": source_title,
                    "filename": pdf_path.name,
                    "page_number": page["page_number"],
                    "chunk_number": chunk_number,
                },
            })

    return chunks
