# CENTROID System & Design Rules
> **Workspace**: `C:\Users\User\Documents\CENTROID\`  
> **Source of Truth**: `AGENT.md`

---

## 1. VISUAL DESIGN SYSTEM TOKENS

When generating, modifying, or styling any HTML, CSS, SVG, or UI component for CENTROID:

### Color Palette
- **Background**: `#14181C` (Deep carbon foundation)
- **Panel Surface**: `#1C2128` (Slate bedrock container)
- **Primary Text**: `#E8E6E1` (Crisp off-white)
- **Secondary Text**: `#8B9199` (Telemetry gray)
- **Hairline Dividers**: `#2A323D` (1px solid, slightly lighter than panel)
- **Accent Brass**: `#C98A3E` (**STRICTLY RESERVED** for the active/selected cluster and live readout values)
- **Secondary Data Accent**: `#5FA8A0` (Cohesion metrics, tags, badges, secondary indicators)

### Typography
- **Display / Headlines / Wordmark**: `Fraunces` (Google Fonts, serif authority)
- **Data / Numbers / Tables / Readout**: `IBM Plex Mono` (Google Fonts, monospace precision)
- **Body Copy**: `IBM Plex Sans` (Google Fonts, sans-serif clarity)

### Geometric & Structural Constraints
- **0px border-radius everywhere** (Strict rectilinear panels, zero rounded corners).
- **0px drop shadows** (Flat planar elevation, no blur/shadows).
- **No gradients** (Pure solid surfaces).
- **Hairline 1px dividers** (`1px solid #2A323D`).
- **Progress Bars**: Pure text-based Unicode progress bars (e.g. `████████░░` or `■■■■■□□□□□`), never HTML `<progress>` elements.
- **Keyboard Focus**: Visible keyboard focus states on all actions (`:focus-visible` with `1px solid #C98A3E` outline/border).

---

## 2. PIPELINE ARCHITECTURE & DATA FLOW

CENTROID processes data sequentially through 6 flat-file stages:

```
[inputs/urls.txt, pdfs/, docs/, raw/]
   ↓ (01_fetch.py + 02_extract_pdf.py)
[outputs/corpus.txt]
   ↓ (03_scan_words.py)
[outputs/word_freq_raw.csv]
   ↓ (Optional: 04_web_expand.py --web-expand)
[outputs/corpus.txt] (updated)
   ↓ (05_cluster.py)
[outputs/word_freq_ranked.csv + outputs/clusters.json]
   ↓ (06_report.py)
[outputs/REPORT.md + outputs/dashboard.html]
```

---

## 3. CORE CONSTRAINTS & DEVELOPMENT RULES

1. **Local-Only Processing**: No cloud calls except initial user URL fetches.
2. **Zero Paid APIs / Keys**: Local NLP and ML using NLTK, Scikit-learn, and Gensim.
3. **Flat File Storage**: Flat files only (`.csv`, `.json`, `.txt`, `.md`, `.html`). No databases.
4. **Standalone Deliverables**: `dashboard_template.html` and `dashboard.html` must open cleanly via `file://` with no build step or web server.
5. **Deterministic Results**: Use `seed=42` / `random_state=42` for all Word2Vec and KMeans models.
6. **Documentation Integrity**: Maintain `AGENT.md` as the central nerve center.
