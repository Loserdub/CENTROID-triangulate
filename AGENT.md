# CENTROID — Agent System File
> **Nerve Center v1.0** | Local · Mechanical · Non-Cloud
> Project root: `C:\Users\User\Documents\CENTROID\`

---

## 1. PROJECT MISSION

CENTROID is a **local word-frequency and semantic cluster analysis engine**.

The user feeds it any combination of:
- Google Trends URLs
- Web article URLs
- PDFs / `.txt` / `.docx` files
- Raw pasted text

CENTROID scans all inputs, extracts every meaningful word, ranks them by frequency and co-occurrence weight, and surfaces **gaps**, **centroids** (high-density word clusters), and **percentage breakdowns** — entirely offline after the initial fetch.

**Core output**: a clean Markdown report with the top N words ranked, organized by semantic cluster, with gap analysis for the user to act on.

---

## 2. FOLDER STRUCTURE

```
CENTROID/
├── AGENT.md                  ← You are here (nerve center)
├── inputs/
│   ├── urls.txt              ← One URL per line (articles, trends, etc.)
│   ├── pdfs/                 ← Drop PDF files here
│   ├── docs/                 ← Drop .txt / .docx files here
│   └── raw/                  ← Paste raw text as .txt files
├── pipeline/
│   ├── 01_fetch.py           ← Fetch/scrape URLs → raw text
│   ├── 02_extract_pdf.py     ← PDF → raw text (PyMuPDF)
│   ├── 03_scan_words.py      ← Word scanner: tokenize, filter, count
│   ├── 04_web_expand.py      ← Optional: web search for related terms
│   ├── 05_cluster.py         ← Frequency ranking + centroid clustering
│   └── 06_report.py          ← Generate Markdown report + interactive HTML dashboard
├── wordlists/
│   ├── stopwords_en.txt      ← Words to always ignore (the, and, is...)
│   ├── target_words.txt      ← Optional: specific words to always track
│   └── custom_ignore.txt     ← User-defined words to exclude
├── outputs/
│   ├── word_freq_raw.csv     ← Every word + raw count
│   ├── word_freq_ranked.csv  ← Top 1000 words sorted by score
│   ├── clusters.json         ← Centroid clusters (semantic groups)
│   ├── REPORT.md             ← Final human-readable report
│   ├── dashboard_template.html ← Standalone interactive dashboard template
│   └── dashboard.html        ← Live populated interactive dashboard
├── requirements.txt          ← Python dependencies
└── run.py                    ← Single entry point: runs full pipeline
```

---

## 3. TECH STACK (Local / Non-Cloud)

| Purpose              | Library                  | Notes                            |
|----------------------|--------------------------|----------------------------------|
| URL scraping         | requests + beautifulsoup4| Static pages; no JS needed       |
| JS-heavy pages       | playwright (optional)    | Headless Chromium, local         |
| PDF extraction       | pymupdf (fitz)           | Fast, local PDF text extraction  |
| DOCX extraction      | python-docx              | Local Word doc parsing           |
| NLP / tokenization   | nltk                     | Stopwords, tokenizer, stemmer    |
| Frequency ranking    | collections.Counter      | Built-in, zero dependencies      |
| Clustering           | scikit-learn (KMeans)    | Local ML, no API needed          |
| Word vectors         | gensim (Word2Vec)        | Train on corpus locally          |
| Web word expand      | duckduckgo-search        | No API key required              |
| Report output        | plain Python f-strings   | No external templating           |
| CLI interface        | argparse                 | Simple command-line runner       |

**Language**: Python 3.11+
**No cloud APIs. No OpenAI. No keys required.**

---

## 4. PIPELINE STAGES

### Stage 1 — INGEST (01_fetch.py, 02_extract_pdf.py)
- Read inputs/urls.txt → HTTP GET each URL → strip HTML → save plain text
- Read all files in inputs/pdfs/ → extract text with PyMuPDF
- Read all .txt / .docx in inputs/docs/
- All raw text saved to a unified corpus.txt in outputs/

### Stage 2 — WORD SCAN (03_scan_words.py)
- Tokenize entire corpus (word boundaries, lowercase)
- Strip punctuation, numbers (configurable)
- Remove stopwords (wordlists/stopwords_en.txt + custom_ignore.txt)
- Optional: stem or lemmatize (NLTK Porter Stemmer)
- Check against target_words.txt — flag those regardless of frequency
- Output: word_freq_raw.csv → columns: word, count, source_file, tfidf_score

### Stage 3 — WEB EXPAND (optional, 04_web_expand.py)
- Take top 50 words from Stage 2
- Run DuckDuckGo searches for each → scrape top 3 results
- Extract new vocabulary from those pages
- Merge into corpus → re-run Stage 2 counts
- Flag "web-sourced" words with a `web` tag in output

### Stage 4 — RANK + CLUSTER (05_cluster.py)
- Sort by combined score: (raw_count × 0.6) + (tfidf_score × 0.4)
- Take top 1000 words
- Build co-occurrence matrix (sliding window of 5 words)
- Train Word2Vec on corpus (local, 100 dimensions)
- Run KMeans (default k=10 clusters) on word vectors
- Label clusters by highest-frequency anchor word
- Output: word_freq_ranked.csv + clusters.json

### Stage 5 — REPORT & DASHBOARD (06_report.py)
- Generate `outputs/REPORT.md` with sections:
  1. Summary — total words scanned, unique words, sources used
  2. Top 1000 Words — ranked table (rank, word, count, %, cluster)
  3. Centroid Clusters — each cluster named + top 20 words
  4. Gap Analysis — words in target_words.txt with LOW frequency
  5. Percentage Breakdown — cluster-level % share of total corpus
- Generate `outputs/dashboard.html` (or standalone `outputs/dashboard_template.html`):
  1. Self-contained single file opening directly via `file://` with no server
  2. Interactive SVG semantic topography cluster map (id="cluster-map")
  3. Live readout panel (id="readout-panel") with dynamic text-based unicode strength bar
  4. Collapsible ranked word ledger table shell (top 1000)

---

## 5. WORD SCANNER RULES

```
RULE 1: Lowercase everything
RULE 2: Remove tokens < 3 characters
RULE 3: Remove tokens matching [^a-z\-'] (numbers, symbols)
RULE 4: Strip stopwords (EN + custom list)
RULE 5: If target_words.txt exists, flag matches before any filtering
RULE 6: Optional stemming (off by default; enable via --stem flag)
RULE 7: Count frequency per source file AND globally
RULE 8: Compute TF-IDF across all source documents
RULE 9: Export raw CSV with all surviving tokens
```

---

## 6. DESIGN SYSTEM SPECIFICATION (Dashboard & Web Outputs)

```
================================================================================
CENTROID VISUAL DESIGN SYSTEM — EXACT TOKENS & CONSTRAINTS
================================================================================
1. COLOR PALETTE:
   - Background:             #14181C (deep carbon foundation)
   - Panel Surface:          #1C2128 (slate bedrock container)
   - Primary Text:           #E8E6E1 (crisp off-white)
   - Secondary Text:         #8B9199 (subdued telemetry gray)
   - Hairline Dividers:      #2A323D (1px solid, slightly lighter than panel)
   - Accent Brass:           #C98A3E (STRICTLY reserved for active/selected cluster & live readout)
   - Secondary Data Accent:  #5FA8A0 (cohesion metrics, tags, badges)

2. TYPOGRAPHY (Loaded via Google Fonts):
   - Display / Headline:     Fraunces (Wordmark, primary headings, serif authority)
   - Data / Labels / Numbers:IBM Plex Mono (Values, tables, telemetry, tags, readouts)
   - Body Copy:              IBM Plex Sans (Descriptions, helper text, explanations)

3. GEOMETRY & STYLING RULES:
   - 0px border-radius everywhere (strict rectilinear panels, zero rounded corners)
   - 0px drop shadows (flat planar elevation)
   - No gradients (pure solid surfaces)
   - 1px hairline borders and dividers (#2A323D)
   - Visible keyboard focus states on all actions (:focus-visible outline: 1px solid #C98A3E)

4. READOUT & DATA PRESENTATION:
   - Text-based Unicode progress bars (e.g. '████████░░' / '■■■■■□□□□□'), NEVER <progress>
   - Two-column grid layout: Left ~70% Cluster Map (#cluster-map), Right ~30% Readout (#readout-panel)
   - Collapsible Ranked Word Ledger table shell below grid
   - Fully responsive down to single-column on narrow viewports
   - Self-contained single-file architecture (must open cleanly via file:// with zero build/server)
================================================================================
```

---

## 7. REPORT OUTPUT FORMAT (REPORT.md skeleton)

```
# CENTROID Report — {date}
Sources scanned: {n} | Total tokens: {n} | Unique words: {n}

## Top 1000 Words
| Rank | Word | Count | % of Corpus | Cluster |

## Centroid Clusters
### Cluster 1 — "{anchor word}"
> Top words: word1, word2 ... word20

## Gap Analysis
Words in your target list with LOW frequency (potential content gaps):
- `target_word` — appeared only {n} times ({x}%)

## Cluster % Breakdown
| Cluster | Word Count | % Share |
```

---

## 8. ENTRY POINT (run.py)

```
# Full pipeline (Ingest → Scan → [Web Expand] → Cluster → Report + Dashboard)
python run.py

# With options
python run.py --top 500 --clusters 8 --stem --web-expand

# Output only (skip re-fetch)
python run.py --report-only
```

Flags:
| Flag            | Default | Description                              |
|-----------------|---------|------------------------------------------|
| --top N         | 1000    | Number of top words in report/dashboard  |
| --clusters N    | 10      | Number of KMeans clusters                |
| --stem          | off     | Enable Porter stemmer                    |
| --web-expand    | off     | Enable DuckDuckGo web expansion          |
| --report-only   | off     | Skip ingest/scan, regenerate report only |
| --lang          | en      | Language for stopwords                   |

---

## 9. DEPENDENCIES (requirements.txt)

```
requests>=2.31.0
beautifulsoup4>=4.12.0
pymupdf>=1.23.0
python-docx>=1.1.0
nltk>=3.8.1
scikit-learn>=1.4.0
gensim>=4.3.2
duckduckgo-search>=5.3.0
```

Install:
```
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger
```

---

## 10. IMMEDIATE NEXT STEPS

- [x] Build pipeline/01_fetch.py through pipeline/05_cluster.py
- [x] Create outputs/dashboard_template.html with strict design system
- [ ] Integrate dashboard generator into pipeline/06_report.py to emit outputs/dashboard.html
- [ ] Wire live cluster data from clusters.json & word_freq_ranked.csv into dashboard.html
- [ ] Verify full run: python run.py → review outputs/REPORT.md and outputs/dashboard.html

---

## 11. AGENT CONSTRAINTS & DEVELOPMENT RULES

```
RULE 1: Local-only CPU/RAM processing — No cloud calls except initial user URL fetches.
RULE 2: Zero paid APIs / keys — Entire NLP & ML stack runs locally with NLTK, Scikit-learn, Gensim.
RULE 3: Flat file architecture — No persistent database; flat files only (CSV, JSON, TXT, MD, HTML).
RULE 4: Strict Design System fidelity — All UI work must adhere exactly to Section 6 tokens.
RULE 5: Standalone HTML deliverables — Dashboard outputs must open directly via file:// with zero build step.
RULE 6: Reproducible determinism — Set random_state=42 for all clustering/W2V models.
RULE 7: Documentation integrity — Always keep AGENT.md updated as the system backbone and source of truth.
```

---
*CENTROID AGENT.md — v1.1 — Updated 2026-09-06*
