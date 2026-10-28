
"""Inspect the expected NICE pages immediately after ingestion."""

from src.ingest_guidelines import ingest_registered_guidelines

TARGET_PAGES = {7, 12}
TARGET_SOURCE = "nice_ng111"


def main():
    chunks, report = ingest_registered_guidelines()

    print("Total chunks:", len(chunks))
    print("\nNICE ingestion report:")

    for item in report:
        if item.get("source_id") == TARGET_SOURCE:
            print(item)

    for page_number in sorted(TARGET_PAGES):
        matches = [
            chunk
            for chunk in chunks
            if chunk.get("metadata", {}).get("source_id") == TARGET_SOURCE
            and chunk.get("metadata", {}).get("page_number") == page_number
        ]

        print("\n" + "=" * 70)
        print(f"NICE PHYSICAL PAGE {page_number}")
        print("Chunks found:", len(matches))

        for chunk in matches:
            print("\nChunk ID:", chunk.get("chunk_id"))
            print("Metadata:", chunk.get("metadata"))
            print("Text preview:", chunk.get("text", "")[:1200])


if __name__ == "__main__":
    main()
