"""Source registry: one record per guideline document, with provenance and limits.
Unverified fields are marked TO_VERIFY and reported, never silently accepted."""

import hashlib
import json
from pathlib import Path

from src.config import PROJECT_ROOT

GUIDELINES_DIR = PROJECT_ROOT / "data" / "guidelines"
REGISTRY_PATH = GUIDELINES_DIR / "source_registry.json"
PLACEHOLDER = "TO_VERIFY"

REQUIRED_FIELDS = [
    "id", "title", "issuing_organization", "source_url", "publication_date",
    "accessed_date", "version_notes", "scope", "exclusions", "license_notes",
    "local_filename", "sha256", "limitations",
]


def load_registry(path: Path = REGISTRY_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Registry must be a non-empty JSON list")
    return data


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_source(record: dict, guidelines_dir: Path = GUIDELINES_DIR) -> dict:
    """Return {'errors': [...], 'unverified': [...]}.
    errors: structural problems, missing file, hash mismatch.
    unverified: fields still empty or marked TO_VERIFY."""
    errors, unverified = [], []
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"missing field: {field}")
        elif record[field] in (None, "", PLACEHOLDER):
            unverified.append(field)

    filename = record.get("local_filename")
    if filename and filename != PLACEHOLDER:
        pdf = Path(guidelines_dir) / filename
        if not pdf.is_file():
            errors.append(f"file not found: {pdf}")
        else:
            recorded = record.get("sha256")
            if( recorded and recorded != PLACEHOLDER and sha256_of(pdf) != recorded.strip().lower()):
                errors.append("sha256 mismatch: file differs from the recorded version")
    return {"errors": errors, "unverified": unverified}


def main() -> int:
    exit_code = 0
    for record in load_registry():
        result = check_source(record)
        print(f"\n[{record.get('id', '?')}]")
        for e in result["errors"]:
            print(f"  ERROR: {e}")
            exit_code = 1
        for u in result["unverified"]:
            print(f"  unverified: {u}")
        if not result["errors"] and not result["unverified"]:
            print("  OK: fully verified")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())