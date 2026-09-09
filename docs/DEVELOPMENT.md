# CENTROID Developer Guide

This guide outlines setup, testing, execution, debugging, and development workflows for engineers working on the CENTROID codebase.

---

## 1. Prerequisites

- **Python 3.11+** (Windows, macOS, or Linux)
- Standard modern web browser (Edge, Chrome, Firefox, Safari) for dashboard inspection.

---

## 2. Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/CENTROID.git
cd CENTROID
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger
```

---

## 3. Running Tests

CENTROID provides two complementary test suites:

### Unit & Regression Test Suite
Tests shared utilities, Unicode punctuation normalization, 9-rule scanner tokenization, clustering determinism, XSS escaping, and state caching:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### End-to-End Integration Smoke Test
Backs up current workspace files, runs the complete pipeline against isolated multi-domain test fixtures (`tests/fixtures/`), and validates all 7 output assertions:
```bash
python tests/validate_pipeline.py
```

---

## 4. Running the Application

### Full Pipeline Run (with Dashboard)
```bash
python run.py --dashboard
```

### Common CLI Flags
```bash
# Set top N words and cluster count (k)
python run.py --top 500 --clusters 8 --dashboard

# Force re-execution of all stages (ignoring cached state)
python run.py --force --dashboard

# Set minimum word threshold for scanning
python run.py --min-words 100 --dashboard

# Enable optional DuckDuckGo web expansion
python run.py --web-expand --dashboard

# Regenerate report and dashboard only from existing cluster data
python run.py --report-only --dashboard
```

### Isolated Stage Execution
Run individual stages using current intermediate outputs in `outputs/`:
```bash
python run.py --stage fetch
python run.py --stage extract
python run.py --stage scan
python run.py --stage cluster
python run.py --stage report
python run.py --stage dashboard
```

---

## 5. Input and Output Locations

### Inputs
- `inputs/urls.txt`: Plain text list of URLs (one per line, `#` comments supported).
- `inputs/pdfs/`: Drop directory for local `.pdf` files.
- `inputs/docs/`: Drop directory for `.docx` and `.txt` documents.
- `inputs/raw/`: Drop directory for raw plain text files.
- `wordlists/stopwords_en.txt`: Curated base English stopwords.
- `wordlists/custom_ignore.txt`: Project-specific or generic conversational exclusions.
- `wordlists/web_coding_lingo.txt`: Web, CSS, JS, and HTML boilerplate terms.
- `wordlists/target_words.txt`: Keywords for Gap Analysis and Directed Mode.

### Outputs
- `outputs/run_state.json`: Stage execution timestamps, exit codes, and SHA-256 hashes.
- `outputs/corpus.txt`: Consolidated, cleaned corpus with source boundary headers.
- `outputs/word_freq_raw.csv`: Raw word counts, source files, and TF-IDF scores.
- `outputs/word_freq_ranked.csv`: Top $N$ words with cluster IDs and combined scores.
- `outputs/clusters.json`: Centroid definitions, anchor words, and percentage share.
- `outputs/REPORT.md`: Markdown intelligence summary.
- `outputs/dashboard_template.html`: Source UI template for the dashboard.
- `outputs/dashboard.html`: Self-contained interactive dashboard (open via `file://`).

---

## 6. Managing Test Fixtures

Test fixtures reside in `tests/fixtures/` and are intentionally domain-agnostic:
- `article_culinary_gastronomy.txt`
- `article_marine_ecosystems.txt`
- `article_quantum_computing.txt`

To add a new fixture:
1. Place a representative plain text file in `tests/fixtures/<name>.txt`.
2. Verify that it contains at least 50 raw words to meet the default scanner threshold.
3. Run `python tests/validate_pipeline.py` to confirm that the pipeline ingests and clusters the new fixture.

---

## 7. Debugging & Logging Guidance

- **Log Level**: All pipeline stages configure standard Python `logging`. Adjust default logging in individual scripts or orchestrator to `DEBUG` for verbose inspection:
  ```python
  logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
  ```
- **Fetch Errors**: Failed or low-content URLs are recorded to `outputs/fetch_errors.log`.
- **State Inspection**: Inspect `outputs/run_state.json` to verify stage status, composite input hashes, and failure error messages.

---

## 8. Common Development Problems & Solutions

| Issue | Cause | Solution |
|---|---|---|
| `[skip] stage already current` | State manager detected unchanged input files. | Pass `--force` (`-f`) to bypass hash caching. |
| `insufficient_corpus` notice | Ingested corpus has fewer than 50 words. | Add more source text or pass `--min-words <N>` with a lower threshold. |
| Missing NLTK resources | Tokenizer or stopword corpora not downloaded. | Run: `python -m nltk.downloader punkt stopwords averaged_perceptron_tagger` |
| PyMuPDF / docx import error | Virtual environment missing dependencies. | Run: `pip install -r requirements.txt` |
| `outputs/dashboard.html` not generated | `--dashboard` flag was omitted. | Include `--dashboard` flag on `python run.py`. |
