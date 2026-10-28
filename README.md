# ConstraintGuard-ED

**Constraint-Aware Evidence Verification for Emergency-Department Antimicrobial Stewardship**

## 1. Project Overview

ConstraintGuard-ED is a research prototype designed to investigate evidence retrieval and patient-context verification for antimicrobial stewardship questions in emergency-department settings.

The system retrieves passages from registered clinical guidelines, expands retrieval when predefined risk terms or other trigger conditions are detected, checks the completeness and format of patient-context fields, and presents retrieved evidence with source and page references.

**Important:** This is a research prototype, not a clinically validated decision-support system. Its output must not be used as a substitute for professional clinical judgment.

## 2. Research Question

Does constraint-aware, risk-triggered evidence retrieval help identify relevant guideline evidence while making missing patient context more visible, compared with baseline retrieval methods?

The current experiments evaluate retrieval performance. They do not establish improved clinical safety or patient outcomes.

## 3. Main Components

1. **Guideline ingestion:** Extracts text and page information from registered PDF guidelines and verifies source metadata.
2. **Embedding generation:** Uses `sentence-transformers/all-MiniLM-L6-v2` to represent guideline passages and questions.
3. **Vector retrieval:** Uses FAISS similarity search to retrieve relevant passages.
4. **Risk-triggered retrieval:** Expands the number of retrieved passages when predefined trigger conditions are met.
5. **Patient-context verification:** Checks required fields for missing or invalid values and flags cases requiring review.
6. **Evidence-bounded synthesis:** Presents retrieved passages with source and page citations instead of claiming that the system has independently validated a treatment.
7. **Evaluation:** Compares retrieval methods against a small, manually curated benchmark.

## 4. Technology Stack

- Python 3.11
- Sentence Transformers
- FAISS
- TF-IDF and hybrid reciprocal-rank fusion (RRF) for retrieval experiments
- Pytest
- PDF-based clinical guideline corpus

## 5. Setup

Open PowerShell in the project root.

Activate the existing virtual environment if needed:

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
.\.venv\Scripts\Activate.ps1
```

If the virtual environment is located in the parent directory, use the project's existing Python executable instead:

```powershell
..\.venv\Scripts\python.exe --version
```

Install project dependencies using the repository's dependency file, if present:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The commands above assume the virtual environment and dependency file are in the locations shown. Adjust paths if your repository layout differs.

## 6. Run the Prototype

From the inner project directory:

```powershell
..\.venv\Scripts\python.exe -m src.constraintguard_pipeline
```

Enter a clinical research question and the requested patient-context fields when prompted.

The program displays retrieved evidence, source and page references, retrieval status, context-check status, and limitations.

## 7. Run Tests

```powershell
..\.venv\Scripts\python.exe -m pytest -v
```

**Recorded result:** 67 tests passed in 10.43 seconds in the development environment on 28 October 2026.

## 8. Run the Evaluation

```powershell
..\.venv\Scripts\python.exe -m scripts.evaluate_constraintguard
```

The detailed evaluation report is saved to:

`data/evaluation/constraintguard_evaluation_results.json`

The benchmark cases are stored in:

`data/evaluation/constraintguard_cases.json`

## 9. Preliminary Results

| Evaluation measure | Observed result |
|---|---:|
| Primary benchmark source-and-page hit rate | 3/5 (60%) |
| Primary mean reciprocal rank (MRR) | 0.600 |
| Mean rank among successful primary matches | 1.0 |
| Supporting evidence hit rate | 1/1 (100%) |
| Diagnostic wording variants retrieved | 2/2 |
| Automated tests passed | 67/67 |

A separate retrieval experiment compared dense, TF-IDF, and hybrid RRF retrieval. On the five primary questions, all three achieved a Hit@5 of 0.60; dense and hybrid RRF achieved an MRR of 0.600, while TF-IDF achieved 0.440.

Two manually authored query-expansion variants were evaluated separately. Their results are exploratory and must not be combined with the primary benchmark.

## 10. Limitations

- The primary evaluation contains only five questions.
- Retrieval performance depends on question wording and the guideline corpus.
- Risk triggers are predefined keywords, not a validated clinical risk classifier.
- Expanding retrieval does not guarantee more relevant evidence.
- Patient-context checks assess field completeness and accepted formats, not treatment appropriateness or clinical safety.
- Retrieved passages may omit important population-specific context.
- Different guidelines may apply to different populations or clinical circumstances.
- The system has not undergone clinical validation or prospective evaluation.

## 11. Future Work

- Improve retrieval of evidence that includes the correct population context.
- Evaluate a larger, independently reviewed benchmark.
- Measure citation accuracy, evidence relevance, and unsupported-answer rates.
- Compare baseline and constraint-aware methods under the same evaluation conditions.
- Improve handling of unknown, invalid, and conflicting patient-context fields.
- Conduct expert review before considering any clinical deployment.

## 12. Research Disclaimer

ConstraintGuard-ED is an experimental academic research prototype. It does not provide validated medical advice, prescribe antibiotics, or replace clinician assessment. All outputs require appropriate human review.
