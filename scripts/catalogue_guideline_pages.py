from pathlib import Path
from src.document_processing import extract_pdf_pages

project_root = Path.cwd()
guidelines_dir = project_root / "data" / "guidelines"
output_dir = project_root / "data" / "evaluation"
output_dir.mkdir(parents=True, exist_ok=True)

pdf_files = sorted(guidelines_dir.glob("*.pdf"))

if not pdf_files:
    raise SystemExit(f"No PDF files found in {guidelines_dir}")

full_output = []
summary_output = []

for pdf_path in pdf_files:
    pages = extract_pdf_pages(pdf_path)
    full_output.append(f"\n{'=' * 90}\nPDF: {pdf_path.name}\n{'=' * 90}\n")
    summary_output.append(f"\n{'=' * 90}\nPDF: {pdf_path.name}\n{'=' * 90}\n")

    for page in pages:
        page_number = page["page_number"]
        text = page["text"]

        full_output.append(
            f"\n--- PDF page {page_number} ---\n{text}\n"
        )

        snippet = " ".join(text.split())
        if len(snippet) > 450:
            snippet = snippet[:450] + "..."

        summary_output.append(
            f"\n[PDF page {page_number}]\n{snippet}\n"
        )

    print(f"Extracted {len(pages)} text-containing pages: {pdf_path.name}")

full_path = output_dir / "guideline_page_catalogue_full.txt"
summary_path = output_dir / "guideline_page_catalogue_summary.txt"

full_path.write_text("".join(full_output), encoding="utf-8")
summary_path.write_text("".join(summary_output), encoding="utf-8")

print("\nCatalogue created successfully.")
print(f"Full text: {full_path}")
print(f"Page summaries: {summary_path}")
