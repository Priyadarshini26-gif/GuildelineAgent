from src.answer_synthesis import synthesize_answer


def test_synthesis_includes_source_and_page_citation():
    response = {
        "question": "What does the guideline recommend?",
        "status": "evidence_retrieved",
        "evidence": [
            {
                "rank": 1,
                "text": "Collect a urine sample before starting antibiotics.",
                "source_id": "nice_ng111",
                "source_title": "NICE NG111",
                "page_number": 6,
                "chunk_id": "nice-ng111-p0006-c001",
            }
        ],
    }

    result = synthesize_answer(response)

    assert result["status"] == "answered"
    assert "Collect a urine sample" in result["answer"]
    assert "[1]" in result["answer"]
    assert result["citations"][0]["source_id"] == "nice_ng111"
    assert result["citations"][0]["page_number"] == 6


def test_synthesis_abstains_when_evidence_is_missing():
    response = {
        "question": "What treatment is recommended?",
        "status": "no_evidence_retrieved",
        "evidence": [],
    }

    result = synthesize_answer(response)

    assert result["status"] == "insufficient_evidence"
    assert result["citations"] == []
    assert "no usable guideline passages" in result["answer"]


def test_synthesis_rejects_empty_question():
    response = {
        "question": "   ",
        "evidence": [],
    }

    try:
        synthesize_answer(response)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for an empty question.")


def test_synthesis_preserves_multiple_citations():
    response = {
        "question": "What does the evidence say?",
        "evidence": [
            {
                "text": "First guideline passage.",
                "source_id": "source_a",
                "source_title": "Guideline A",
                "page_number": 4,
                "chunk_id": "chunk_a",
            },
            {
                "text": "Second guideline passage.",
                "source_id": "source_b",
                "source_title": "Guideline B",
                "page_number": 9,
                "chunk_id": "chunk_b",
            },
        ],
    }

    result = synthesize_answer(response)

    assert len(result["citations"]) == 2
    assert "[1]" in result["answer"]
    assert "[2]" in result["answer"]
    assert result["citations"][1]["source_id"] == "source_b"