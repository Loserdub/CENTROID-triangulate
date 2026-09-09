#!/usr/bin/env python3
"""
CENTROID Pipeline — Stage 5/6: Report Generator
Specification: AGENT.md Section 6 (REPORT OUTPUT FORMAT) & Section 4 Stage 5

Description:
  1. Reads `outputs/word_freq_ranked.csv` and `outputs/clusters.json`.
  2. Cross-references `wordlists/target_words.txt` for Gap Analysis:
     Flags any target word in the bottom third of frequency (or not at all) as a gap.
  3. Computes Cluster % Breakdown:
     Each cluster's unique word count as a percentage of total unique words counted.
  4. Generates `outputs/REPORT.md` adhering strictly to AGENT.md section 6 skeleton:
     - Summary (date, sources scanned, total tokens, unique words)
     - Top 1000 Words table (Rank, Word, Count, % of Corpus, Cluster)
     - Centroid Clusters (top 20 words each)
     - Gap Analysis (potential content gaps)
     - Cluster % Breakdown (Cluster, Word Count, % Share)

Usage:
  python pipeline/06_report.py
  python pipeline/06_report.py --top 500
  python pipeline/06_report.py --ranked outputs/word_freq_ranked.csv --clusters outputs/clusters.json
"""

import argparse
import csv
from datetime import date
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Base directory is the project root (CENTROID/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Ensure project root is in sys.path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from pipeline.common import load_ranked_words_from_csv, load_wordlist
except ImportError:
    from common import load_ranked_words_from_csv, load_wordlist

# Default file paths
DEFAULT_RANKED_CSV = BASE_DIR / "outputs" / "word_freq_ranked.csv"
DEFAULT_CLUSTERS_JSON = BASE_DIR / "outputs" / "clusters.json"
DEFAULT_RAW_CSV = BASE_DIR / "outputs" / "word_freq_raw.csv"
DEFAULT_CORPUS_FILE = BASE_DIR / "outputs" / "corpus.txt"
DEFAULT_TARGET_WORDS_FILE = BASE_DIR / "wordlists" / "target_words.txt"
DEFAULT_REPORT_FILE = BASE_DIR / "outputs" / "REPORT.md"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("06_report")


def load_ranked_words(ranked_csv_path: Path, top_n: int = 1000) -> List[Dict[str, Any]]:
    """
    Loads ranked word frequencies from outputs/word_freq_ranked.csv.
    """
    ranked_words = load_ranked_words_from_csv(ranked_csv_path, top_n=top_n)
    logger.info(f"Loaded {len(ranked_words):,} ranked words from '{ranked_csv_path}'.")
    return ranked_words


def load_clusters(clusters_json_path: Path) -> Dict[str, Any]:
    """
    Loads cluster definitions from `outputs/clusters.json`.
    Expected structure:
      {
        "cluster_id": {
          "label": str,
          "words": List[str],
          "total_count": int,
          "pct_share": float
        }
      }
    """
    if not clusters_json_path.exists():
        logger.warning(f"Clusters JSON file not found: {clusters_json_path}")
        return {}

    try:
        with open(clusters_json_path, mode="r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
            if isinstance(data, dict):
                logger.info(f"Loaded {len(data)} clusters from '{clusters_json_path}'.")
                return data
            logger.warning(f"Unexpected data format in '{clusters_json_path}'.")
            return {}
    except Exception as e:
        logger.error(f"Failed to load clusters JSON from '{clusters_json_path}': {e}")
        return {}


def load_corpus_stats(
    raw_csv_path: Path,
    corpus_path: Path,
    ranked_words: List[Dict[str, Any]]
) -> Tuple[int, int, int, Dict[str, int]]:
    """
    Computes corpus-level statistics:
      - sources_count: Number of distinct sources scanned
      - total_tokens: Total token occurrences
      - unique_words_count: Total unique vocabulary count
      - all_word_counts: Dict mapping word -> total_count across corpus
    """
    sources: Set[str] = set()
    all_word_counts: Dict[str, int] = {}
    total_tokens = 0

    # 1. Parse raw CSV if present
    if raw_csv_path.exists():
        try:
            with open(raw_csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row or not row.get("word"):
                        continue
                    word = row["word"].strip().lower()
                    try:
                        count = int(float(row.get("count", 1)))
                    except (ValueError, TypeError):
                        count = 1
                    source = row.get("source_file", "").strip()
                    if source:
                        sources.add(source)
                    all_word_counts[word] = all_word_counts.get(word, 0) + count
                    total_tokens += count
        except Exception as e:
            logger.warning(f"Error reading raw CSV '{raw_csv_path}': {e}")

    # 2. If sources set is empty, check corpus.txt boundary markers
    if not sources and corpus_path.exists():
        try:
            with open(corpus_path, mode="r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                marker_pattern = re.compile(r"(?m)^[-=]{3,}\s*SOURCE:\s*(.*?)\s*[-=]{3,}\s*$")
                for match in marker_pattern.finditer(content):
                    src_line = match.group(1).strip()
                    if src_line:
                        sources.add(src_line)
        except Exception as e:
            logger.warning(f"Error reading corpus markers from '{corpus_path}': {e}")

    # 3. Fallbacks if raw CSV was not populated
    if not all_word_counts and ranked_words:
        for rw in ranked_words:
            w = rw["word"]
            c = rw["count"]
            all_word_counts[w] = c
            total_tokens += c

    unique_words_count = len(all_word_counts)
    sources_count = len(sources) if sources else (1 if corpus_path.exists() else 0)

    if total_tokens == 0 and ranked_words:
        total_tokens = sum(rw["count"] for rw in ranked_words)

    logger.info(
        f"Corpus stats: {sources_count} source(s), {total_tokens:,} total tokens, {unique_words_count:,} unique words."
    )
    return sources_count, total_tokens, unique_words_count, all_word_counts


def load_target_words(target_words_path: Path) -> List[str]:
    """
    Loads target words from `wordlists/target_words.txt`.
    Ignores comments (#) and blank lines.
    """
    if not target_words_path.exists():
        logger.debug(f"Target words file not found: {target_words_path}")
        return []

    target_words: List[str] = []
    seen: Set[str] = set()
    try:
        with open(target_words_path, mode="r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                cleaned = line.strip().lower()
                if cleaned and not cleaned.startswith("#") and cleaned not in seen:
                    target_words.append(cleaned)
                    seen.add(cleaned)
    except Exception as e:
        logger.warning(f"Error reading target words from '{target_words_path}': {e}")

    logger.info(f"Loaded {len(target_words)} target word(s) from '{target_words_path}'.")
    return target_words


def perform_gap_analysis(
    target_words: List[str],
    all_word_counts: Dict[str, int],
    total_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Gap Analysis:
      Cross-references `wordlists/target_words.txt` against final counts.
      Flags any target word appearing in the bottom third of frequency (or not at all) as a gap.

    Returns:
      List of dicts: [{'word': str, 'count': int, 'pct': float, 'in_corpus': bool, 'reason': str}, ...]
    """
    if not target_words:
        return []

    # Sort all unique words by count descending to establish frequency distribution
    sorted_words = sorted(all_word_counts.items(), key=lambda item: item[1], reverse=True)
    total_unique = len(sorted_words)

    # Calculate bottom third cutoff
    # Top third: ranks 1 to floor(total_unique / 3)
    # Middle third: ranks floor(total_unique / 3) + 1 to floor(2 * total_unique / 3)
    # Bottom third: ranks > floor(2 * total_unique / 3)
    cutoff_rank = (2 * total_unique) // 3
    cutoff_count = sorted_words[cutoff_rank][1] if (sorted_words and cutoff_rank < total_unique) else 1

    word_rank_map = {word: rank for rank, (word, _) in enumerate(sorted_words, start=1)}

    gaps: List[Dict[str, Any]] = []

    for target_word in target_words:
        count = all_word_counts.get(target_word, 0)
        pct = (count / total_tokens * 100) if total_tokens > 0 else 0.0

        if count == 0:
            # Word does not appear at all in corpus
            gaps.append({
                "word": target_word,
                "count": 0,
                "pct": 0.0,
                "in_corpus": False,
                "reason": "not present in corpus"
            })
        else:
            rank = word_rank_map.get(target_word, total_unique)
            # Flag if rank is in bottom third or count <= cutoff count
            if rank > cutoff_rank or count <= cutoff_count:
                gaps.append({
                    "word": target_word,
                    "count": count,
                    "pct": pct,
                    "in_corpus": True,
                    "reason": f"bottom third frequency (rank {rank:,}/{total_unique:,})"
                })

    logger.info(f"Gap Analysis: identified {len(gaps)} potential content gap(s) out of {len(target_words)} target word(s).")
    return gaps


def build_markdown_report(
    report_date_str: str,
    sources_count: int,
    total_tokens: int,
    unique_words_count: int,
    ranked_words: List[Dict[str, Any]],
    clusters_data: Dict[str, Any],
    gap_analysis_results: List[Dict[str, Any]],
    target_words_present: bool,
    top_n: int = 1000
) -> str:
    """
    Constructs the exact Markdown report conforming to AGENT.md section 6 skeleton:

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
    """
    lines: List[str] = []

    # 1. Header and Summary
    lines.append(f"# CENTROID Report — {report_date_str}")
    lines.append(f"Sources scanned: {sources_count:,} | Total tokens: {total_tokens:,} | Unique words: {unique_words_count:,}")
    lines.append("")

    # 2. Top 1000 Words Table
    table_title = f"Top {top_n} Words" if top_n == 1000 else f"Top {len(ranked_words)} Words"
    lines.append(f"## {table_title}")
    lines.append("| Rank | Word | Count | % of Corpus | Cluster |")
    lines.append("|---|---|---|---|---|")

    for entry in ranked_words:
        rank = entry["rank"]
        word = entry["word"]
        count = entry["count"]
        pct_of_corpus = (count / total_tokens * 100) if total_tokens > 0 else 0.0
        cluster_label = entry.get("cluster_label", "unclustered")
        lines.append(f"| {rank} | {word} | {count:,} | {pct_of_corpus:.2f}% | {cluster_label} |")

    lines.append("")

    # 3. Centroid Clusters (Top 20 words each)
    lines.append("## Centroid Clusters")

    # Sort clusters by total count descending
    sorted_cluster_entries = sorted(
        clusters_data.items(),
        key=lambda item: item[1].get("total_count", 0),
        reverse=True
    )

    if not sorted_cluster_entries:
        lines.append("*No clusters available.*")
    else:
        for idx, (cid, cdata) in enumerate(sorted_cluster_entries, start=1):
            anchor_word = cdata.get("label", f"Cluster {idx}")
            words_list = cdata.get("words", [])
            top_20_words = words_list[:20]
            words_str = ", ".join(top_20_words)

            lines.append(f"### Cluster {idx} — \"{anchor_word}\"")
            lines.append(f"> Top words: {words_str}")
            lines.append("")

    # 4. Gap Analysis
    lines.append("## Gap Analysis")
    lines.append("Words in your target list with LOW frequency (potential content gaps):")

    if not target_words_present:
        lines.append("- *(No target words defined in `wordlists/target_words.txt`)*")
    elif not gap_analysis_results:
        lines.append("- *(No content gaps detected — all target words are well-represented in the corpus)*")
    else:
        for gap in gap_analysis_results:
            w = gap["word"]
            cnt = gap["count"]
            pct = gap["pct"]
            lines.append(f"- `{w}` — appeared only {cnt:,} times ({pct:.2f}%)")

    lines.append("")

    # 5. Cluster % Breakdown
    # Requirement: each cluster's word count as a percentage of total unique words counted
    lines.append("## Cluster % Breakdown")
    lines.append("| Cluster | Word Count | % Share |")
    lines.append("|---|---|---|")

    # Calculate total unique words clustered
    total_unique_words_clustered = sum(len(cdata.get("words", [])) for _, cdata in sorted_cluster_entries)
    if total_unique_words_clustered == 0:
        total_unique_words_clustered = len(ranked_words) if ranked_words else 1

    for idx, (cid, cdata) in enumerate(sorted_cluster_entries, start=1):
        anchor_word = cdata.get("label", f"Cluster {idx}")
        cluster_unique_count = len(cdata.get("words", []))
        pct_share = (cluster_unique_count / total_unique_words_clustered * 100) if total_unique_words_clustered > 0 else 0.0

        cluster_name_display = f"Cluster {idx} — \"{anchor_word}\""
        lines.append(f"| {cluster_name_display} | {cluster_unique_count:,} | {pct_share:.2f}% |")

    lines.append("")
    return "\n".join(lines)


def generate_report(
    ranked_csv_path: Optional[Path] = None,
    clusters_json_path: Optional[Path] = None,
    raw_csv_path: Optional[Path] = None,
    corpus_path: Optional[Path] = None,
    target_words_path: Optional[Path] = None,
    output_report_path: Optional[Path] = None,
    top_n: int = 1000,
    report_date_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main orchestration function for Stage 6 Report Generation:
      1. Loads ranked words and cluster definitions.
      2. Computes summary statistics.
      3. Cross-references target words for Gap Analysis.
      4. Computes Cluster % Breakdown.
      5. Writes outputs/REPORT.md matching AGENT.md section 6 skeleton exactly.

    Returns:
      Dict with summary execution information.
    """
    target_ranked_csv = ranked_csv_path or DEFAULT_RANKED_CSV
    target_clusters_json = clusters_json_path or DEFAULT_CLUSTERS_JSON
    target_raw_csv = raw_csv_path or DEFAULT_RAW_CSV
    target_corpus = corpus_path or DEFAULT_CORPUS_FILE
    target_target_words = target_words_path or DEFAULT_TARGET_WORDS_FILE
    target_report_file = output_report_path or DEFAULT_REPORT_FILE
    today_str = report_date_str or date.today().isoformat()

    # Ensure output directory exists
    target_report_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load ranked words and cluster data
    ranked_words = load_ranked_words(target_ranked_csv, top_n=top_n)
    clusters_data = load_clusters(target_clusters_json)

    # 2. Load corpus-level statistics
    sources_count, total_tokens, unique_words_count, all_word_counts = load_corpus_stats(
        raw_csv_path=target_raw_csv,
        corpus_path=target_corpus,
        ranked_words=ranked_words
    )

    # 3. Load target words & perform gap analysis
    target_words = load_target_words(target_target_words)
    gap_analysis_results = perform_gap_analysis(
        target_words=target_words,
        all_word_counts=all_word_counts,
        total_tokens=total_tokens
    )

    # 4. Generate Markdown report content
    report_content = build_markdown_report(
        report_date_str=today_str,
        sources_count=sources_count,
        total_tokens=total_tokens,
        unique_words_count=unique_words_count,
        ranked_words=ranked_words,
        clusters_data=clusters_data,
        gap_analysis_results=gap_analysis_results,
        target_words_present=len(target_words) > 0,
        top_n=top_n
    )

    # 5. Write outputs/REPORT.md
    with open(target_report_file, mode="w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"Successfully generated CENTROID report at '{target_report_file}'.")

    # Console preview summary
    print("\n" + "=" * 64)
    print(f" CENTROID Report Generated - {today_str}")
    print("=" * 64)
    print(f" Output Location:  {target_report_file}")
    print(f" Sources Scanned:  {sources_count:,}")
    print(f" Total Tokens:     {total_tokens:,}")
    print(f" Unique Words:     {unique_words_count:,}")
    print(f" Top Words Ranked: {len(ranked_words):,}")
    print(f" Clusters:         {len(clusters_data)}")
    print(f" Gap Words Flagged:{len(gap_analysis_results):,}")
    print("=" * 64 + "\n")

    return {
        "report_file": str(target_report_file),
        "sources_count": sources_count,
        "total_tokens": total_tokens,
        "unique_words_count": unique_words_count,
        "ranked_words_count": len(ranked_words),
        "clusters_count": len(clusters_data),
        "gaps_count": len(gap_analysis_results),
    }


def main():
    parser = argparse.ArgumentParser(
        description="CENTROID Pipeline Stage 6 — Report Generator: Produce outputs/REPORT.md"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=1000,
        help="Number of top words to include in the report table (default: 1000)."
    )
    parser.add_argument(
        "--ranked",
        type=Path,
        default=DEFAULT_RANKED_CSV,
        help=f"Path to ranked word frequency CSV (default: {DEFAULT_RANKED_CSV})."
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=DEFAULT_CLUSTERS_JSON,
        help=f"Path to clusters JSON file (default: {DEFAULT_CLUSTERS_JSON})."
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=DEFAULT_RAW_CSV,
        help=f"Path to raw word frequency CSV (default: {DEFAULT_RAW_CSV})."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_FILE,
        help=f"Path to corpus text file (default: {DEFAULT_CORPUS_FILE})."
    )
    parser.add_argument(
        "--target-words",
        type=Path,
        default=DEFAULT_TARGET_WORDS_FILE,
        help=f"Path to target words list (default: {DEFAULT_TARGET_WORDS_FILE})."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_FILE,
        help=f"Path to destination REPORT.md (default: {DEFAULT_REPORT_FILE})."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Report header date string (default: today's date YYYY-MM-DD)."
    )

    args = parser.parse_args()
    try:
        generate_report(
            ranked_csv_path=args.ranked,
            clusters_json_path=args.clusters,
            raw_csv_path=args.raw_csv,
            corpus_path=args.corpus,
            target_words_path=args.target_words,
            output_report_path=args.output,
            top_n=args.top,
            report_date_str=args.date
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
