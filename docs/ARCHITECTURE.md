# CENTROID System Architecture

## 1. System Purpose & Core Principles

**CENTROID** is a deterministic, offline natural language processing and semantic cluster analysis workbench. It operates without external LLM tokens, cloud APIs, or telemetry, processing unstructured documents into structured lexical topography on standard CPU and memory.

### Core Principles
1. **Local-First & Offline**: Zero external network calls during analysis. Ingestion can operate completely on local files (`pdfs/`, `docs/`, `raw/`) or static URLs.
2. **Deterministic & Reproducible**: Fixed seeds (`seed=42`, `random_state=42`) and MD5-derived 32-bit positive integer hashes for vocabulary terms guarantee identical output on repeated runs.
3. **Transparent Data Boundaries**: Every stage consumes explicit flat-file inputs and produces explicit flat-file outputs (`.txt`, `.csv`, `.json`, `.md`, `.html`).
4. **Resilient State Management**: Cryptographic SHA-256 caching detects unchanged inputs and skips redundant stages, enabling rapid reruns and partial resumes.

---

## 2. High-Level Architecture Diagram

```
+-----------------------------------------------------------------------------+
|                                INPUT SOURCES                                |
|  - inputs/urls.txt (Web links)            - inputs/docs/ (.docx, .txt)      |
|  - inputs/pdfs/    (PDF dossiers)         - inputs/raw/  (Plaintext files)  |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|               STAGE 00: STATE & CHECKPOINT MANAGER (00_state.py)            |
|  - Computes SHA-256 hashes of input files and parameter payloads            |
|  - Queries outputs/run_state.json; skips current stages; tracks durations   |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                STAGE 01 & 02: INGESTION & EXTRACTION                        |
|  - 01_fetch.py:       Streaming HTTP fetch, Content-Type validation,        |
|                       10MB size limit, boilerplate HTML scrubbing           |
|  - 02_extract_pdf.py: Local PDF (PyMuPDF) & DOCX extraction,                |
|                       Unicode punctuation & line unwrapping                 |
+-----------------------------------------------------------------------------+
                                       |
                                       v  [outputs/corpus.txt]
+-----------------------------------------------------------------------------+
|             STAGE 03: SIGNAL / TERM EXTRACTION (03_scan_words.py)           |
|  - Applies 9 lexical rules: lowercasing, min length 3, [^a-z\-'] pruning,   |
|    stopword scrubbing (EN + custom ignore + coding lingo), target words     |
|  - Computes document-level TF-IDF across source partitions                  |
+-----------------------------------------------------------------------------+
                                       |
                                       v  [outputs/word_freq_raw.csv]
          [Optional: --web-expand] ----+----> STAGE 04: WEB EXPANSION
                                       |      (04_web_expand.py)
                                       |      Rate-limited DuckDuckGo queries;
                                       |      appends terms to corpus.txt
                                       v
+-----------------------------------------------------------------------------+
|            STAGE 05: VECTOR EMBEDDING & CENTROID CLUSTERING                 |
|                               (05_cluster.py)                               |
|  1. Signal Ranking: combined_score = (0.60 * count) + (0.40 * tfidf_score) |
|  2. Co-occurrence: 5-word sliding window co-occurrence matrix               |
|  3. Representation: Local 100-dim Word2Vec CBOW neural embedding (seed=42)  |
|  4. Centroid Discovery: KMeans clustering (k clusters, random_state=42)     |
|  5. Triangulation: Distance-to-mean calculation, anchor word identification |
+-----------------------------------------------------------------------------+
                                       |
                    +------------------+------------------+
                    |                                     |
                    v [outputs/clusters.json]             v [outputs/word_freq_ranked.csv]
+---------------------------------------+   +---------------------------------------+
|      STAGE 06: INTELLIGENCE REPORT    |   |      STAGE 07: INTERACTIVE WORKBENCH  |
|            (06_report.py)             |   |           (07_dashboard.py)           |
|  - Lexical gap analysis against       |   |  - Interpolates dashboard_template    |
|    wordlists/target_words.txt         |   |  - Sanitized inline JSON payload      |
|  - Cluster % share & token telemetry  |   |  - HTML-escaped tabular records       |
|  - Formatted Markdown summary         |   |  - Standalone single-file HTML        |
+---------------------------------------+   +---------------------------------------+
                    |                                     |
                    v                                     v
           outputs/REPORT.md                     outputs/dashboard.html
```

---

## 3. Analytical Methodology

The CENTROID pipeline implements a deterministic seven-step analytical progression:

```
CORPUS
  --> SIGNAL / TERM EXTRACTION
  --> SIGNAL RANKING
  --> RELATIONSHIP / CO-OCCURRENCE ANALYSIS
  --> TRIANGULATION
  --> SUBJECT / STRUCTURE DISCOVERY
  --> GAP / COVERAGE ANALYSIS
  --> REPORT / DASHBOARD
```

### 1. Corpus Preparation
Unstructured text from disparate file types (web, PDF, DOCX, TXT) is normalized into uniform ASCII-compatible text via `pipeline.common.normalize_unicode_punctuation`. Documents are delimited in `outputs/corpus.txt` with standard source metadata headers (`--- SOURCE: [type] | file/url: [ident] ---`).

### 2. Signal / Term Extraction
`pipeline/03_scan_words.py` tokenizes document text and filters noise using a strict 9-rule scanner:
- Rule 1: Lowercase all text.
- Rule 2: Enforce minimum token length ($\ge 3$ characters).
- Rule 3: Enforce valid word morphology (`^[a-z]+(?:[-'][a-z]+)*$`), rejecting digits and symbols.
- Rule 4: Strip curated English stopwords (`wordlists/stopwords_en.txt`).
- Rule 5: Strip custom user exclusions (`wordlists/custom_ignore.txt`).
- Rule 6: Strip web and UI boilerplate lingo (`wordlists/web_coding_lingo.txt`).
- Rule 7: Preserve target domain words unconditionally (`wordlists/target_words.txt`).
- Rule 8: Compute document-level Term Frequency-Inverse Document Frequency (TF-IDF):
  $$\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \left(1.0 + \ln\left(\frac{N + 1}{\text{DF}(w) + 1}\right)\right)$$
- Rule 9: Export per-source and aggregate counts to `outputs/word_freq_raw.csv`.

### 3. Signal Ranking
In `pipeline/05_cluster.py`, vocabulary terms are ranked using a weighted multi-factor heuristic:
$$\text{Score}(w) = (0.60 \times \text{Raw Count}) + (0.40 \times \text{TF-IDF Score})$$
This balances high-frequency thematic terms with distinctive, content-rich markers.

### 4. Relationship & Co-occurrence Analysis
A symmetric 5-word sliding context window computes pair-wise co-occurrence metrics across corpus sentences. An offline 100-dimensional continuous bag-of-words (CBOW) Word2Vec model is trained locally (`min_count=1`, `window=5`, `seed=42`, `workers=1`). Terms outside the core vocabulary receive deterministic pseudo-vectors seeded via `MD5(word)`.

### 5. Triangulation & Subject / Structure Discovery
Using the resulting dense vector space, Scikit-learn's K-Means algorithm partitions the top $N$ words into $k$ distinct semantic clusters (`random_state=42`). 
- **Anchor Term Identification**: The member word with the highest raw occurrence count serves as the cluster's primary anchor/label.
- **Topological Metrics**: For every cluster, the engine computes Euclidean distance to centroid mean, cluster density, cohesion, and percentage token share of the corpus.

### 6. Gap & Coverage Analysis
`pipeline/06_report.py` and `pipeline/07_dashboard.py` cross-reference the extracted vocabulary against `wordlists/target_words.txt`:
- **ABSENT**: Target words with 0 occurrences in the corpus.
- **LOW FREQ**: Target words present at less than 0.05% of total corpus tokens.
- **PRESENT**: Target words actively represented within cluster structures.

### 7. Reporting & Dashboard Generation
- **`outputs/REPORT.md`**: Executive Markdown summary containing telemetry summaries, ranked tables, cluster distributions, and gap matrices.
- **`outputs/dashboard.html`**: Standalone browser dashboard compiled from `outputs/dashboard_template.html`. Features coordinate-mapped SVG cartography, live telemetry readouts, collapsed tabular ledgers, and an in-browser ingestion bay.

---

## 4. Component Directory & File Responsibilities

| Component / File | Primary Responsibility | Input Boundary | Output Boundary |
|---|---|---|---|
| `run.py` | Root CLI shim; detects `.venv` Python and delegates. | CLI arguments | Subprocess return code |
| `pipeline/run.py` | Pipeline orchestrator, flag parsing, stage timing. | CLI flags | Pipeline console telemetry |
| `pipeline/common.py` | Shared utilities: Unicode normalization, MD5 seeds, HTML parsing, safe JSON. | Raw text / objects | Cleaned tokens / strings |
| `pipeline/00_state.py` | Checkpoint caching, SHA-256 hashing, skip detection. | Input file contents | `outputs/run_state.json` |
| `pipeline/01_fetch.py` | Web scraper; Content-Type check, 10MB limit. | `inputs/urls.txt` | Appends to `outputs/corpus.txt` |
| `pipeline/02_extract_pdf.py` | PDF/DOCX extractor using PyMuPDF and python-docx. | `inputs/pdfs/`, `inputs/docs/` | Appends to `outputs/corpus.txt` |
| `pipeline/03_scan_words.py` | 9-rule scanner, TF-IDF calculation. | `outputs/corpus.txt`, wordlists | `outputs/word_freq_raw.csv` |
| `pipeline/04_web_expand.py` | DuckDuckGo search keyword expansion (optional). | Top 50 words from raw CSV | Appends to `outputs/corpus.txt` |
| `pipeline/05_cluster.py` | Word2Vec embedding, KMeans clustering, anchor detection. | `outputs/word_freq_raw.csv`, corpus | `outputs/clusters.json`, `word_freq_ranked.csv` |
| `pipeline/06_report.py` | Markdown report compilation, gap analysis. | Ranked CSV, clusters JSON | `outputs/REPORT.md` |
| `pipeline/07_dashboard.py` | Compiles HTML dashboard with XSS-safe serialization. | Dashboard template, ranked CSV, clusters JSON | `outputs/dashboard.html` |
| `tests/test_centroid.py` | Comprehensive unit and edge-case test suite. | Mock inputs, temp directories | Test pass/fail assertions |
| `tests/validate_pipeline.py` | End-to-end integration smoke test suite. | `tests/fixtures/*.txt` | Pipeline verification report |
