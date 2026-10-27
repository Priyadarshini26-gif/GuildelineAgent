
from src.answer_generation import prepare_evidence_response


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, question, top_k=5):
        return self.results[:top_k]


def test_returns_retrieved_evidence_with_source_and_page():
    retriever = FakeRetriever([
        {
            "text": "Example guideline passage.",
            "chunk_id": "nice-ng111-p0023-c001",
            "metadata": {
                "source_id": "nice_ng111",
                "source_title": "NICE NG111",
                "page_number": 23,
            },
        }
    ])

    response = prepare_evidence_response(
        retriever,
        "What does the guideline say?",
    )

    assert response["status"] == "evidence_retrieved"
    assert len(response["evidence"]) == 1
    assert response["evidence"][0]["source_id"] == "nice_ng111"
    assert response["evidence"][0]["page_number"] == 23


def test_handles_no_retrieved_evidence():
    retriever = FakeRetriever([])

    response = prepare_evidence_response(
        retriever,
        "What does the guideline say?",
    )

    assert response["status"] == "no_evidence_retrieved"
    assert response["evidence"] == []


def test_rejects_empty_question():
    retriever = FakeRetriever([])

    try:
        prepare_evidence_response(retriever, "   ")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for an empty question.")
