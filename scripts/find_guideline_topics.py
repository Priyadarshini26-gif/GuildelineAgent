from pathlib import Path
import re
from src.document_processing import extract_pdf_pages

guidelines_dir = Path("data/guidelines")

topics = {
    "classification and diagnosis": r"\b(classification|diagnos\w*|urinalysis|urine culture|pyuria)\b",
    "empiric treatment": r"\b(empiric|empirical|initial treatment|antibiotic therapy|antimicrobial therapy)\b",
    "IV-to-oral transition": r"\b(intravenous|oral transition|switch to oral|IV.to.oral|oral antibiotic)\b",
    "treatment duration": r"\b(duration of therapy|treatment duration|days of therapy|duration)\b",
    "pyelonephritis": r"\bpyelonephritis\b",
    "resistance and susceptibility": r"\b(resistan\w*|susceptib\w*|culture results)\b",
}

pdf_files = sorted(guidelines_dir.glob("*.pdf"))

for pdf_path in pdf_files:
    pages = extract_pdf_pages(pdf_path)
    print(f"\n{'=' * 88}\nSOURCE: {pdf_path.name}\n{'=' * 88}")

    for topic, pattern in topics.items():
        matches = []
        for page in pages:
            if re.search(pattern, page["text"], re.IGNORECASE):
                text = " ".join(page["text"].split())
                matches.append((page["page_number"], text))

        if matches:
            print(f"\n  {topic}:")
            for page_number, text in matches:
                print(f"    PDF page {page_number}: {text[:280]}...")

print("\nTopic search complete. These are candidate pages, not verified answer keys.")
