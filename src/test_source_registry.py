from src.source_registry import (
    PLACEHOLDER, REQUIRED_FIELDS, check_source, load_registry, sha256_of,
)


def make_record(**overrides):
    record = {f: "x" for f in REQUIRED_FIELDS}
    record["local_filename"] = "doc.pdf"
    record["sha256"] = PLACEHOLDER
    record.update(overrides)
    return record


def test_real_registry_has_all_fields_and_unique_ids():
    records = load_registry()
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids))
    for r in records:
        for field in REQUIRED_FIELDS:
            assert field in r, f"{r['id']} missing {field}"


def test_placeholder_reported_as_unverified(tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"abc")
    result = check_source(make_record(exclusions=PLACEHOLDER), tmp_path)
    assert "exclusions" in result["unverified"]
    assert result["errors"] == []


def test_missing_field_is_error(tmp_path):
    record = make_record()
    del record["scope"]
    assert any("scope" in e for e in check_source(record, tmp_path)["errors"])


def test_missing_file_is_error(tmp_path):
    result = check_source(make_record(), tmp_path)
    assert any("file not found" in e for e in result["errors"])


def test_hash_mismatch_is_error(tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"abc")
    result = check_source(make_record(sha256="0" * 64), tmp_path)
    assert any("sha256 mismatch" in e for e in result["errors"])


def test_matching_hash_passes(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"abc")
    result = check_source(make_record(sha256=sha256_of(pdf)), tmp_path)
    assert result["errors"] == []

def test_hash_comparison_ignores_case_and_spaces(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"abc")
    messy = "  " + sha256_of(pdf).upper() + " "
    result = check_source(make_record(sha256=messy), tmp_path)
    assert result["errors"] == []