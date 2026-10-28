
"""Check whether the expected NICE pages exist in extracted chunks."""

from src.document_processing import extract_pdf_pages

PDF_PATH = "data/guidelines/nice_ng111.pdf"
TARGET_PAGES = [7, 12]


def main():
    pages = extract_pdf_pages(PDF_PATH)

    for target in TARGET_PAGES:
        print("\n" + "=" * 70)
        print(f"CHECKING PHYSICAL PDF PAGE {target}")
        print("=" * 70)

        matches = [
            page for page in pages
            if page["page_number"] == target
        ]

        if not matches:
            print("PAGE NOT FOUND in extracted pages")
            continue

        for page in matches:
            text = page["text"]
            print("Extracted page number:", page["page_number"])
            print("Text length:", len(text))
            print("Text preview:")
            print(text[:1800])


if __name__ == "__main__":
    main()
