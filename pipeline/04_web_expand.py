#!/usr/bin/env python3
"""
CENTROID Pipeline - Stage 3: Web Expand (Optional)
Specification: AGENT.md Section 4 Stage 3

Description:
  Takes the top N (default 50) words by count from outputs/word_freq_raw.csv,
  queries DuckDuckGo for each word, pulls the top search result URLs (default 3),
  fetches and extracts text from those URLs (reusing 01_fetch.py extraction
  logic if present), and appends the extracted text to outputs/corpus.txt
  with a 'web-expand' source marker.

Requirements:
  - Read outputs/word_freq_raw.csv, take the top 50 words by count
  - For each, run a duckduckgo-search query, pull the top 3 result URLs
  - Fetch and extract text from those URLs the same way 01_fetch.py does
    (reuse its extraction function if present - import it, don't duplicate it)
  - Append this new text to outputs/corpus.txt with a "web-expand" source marker
  - After appending, do NOT re-run scanning here - that's 03's job on the next pass
  - This script only runs when called with --web-expand; make it a standalone
    script that no other stage depends on
  - Rate-limit search calls (1 request per second) to avoid getting blocked

Usage:
  python pipeline/04_web_expand.py --web-expand
  python pipeline/04_web_expand.py --web-expand --top 50 --results-per-query 3 --rate-limit 1.0
"""

import argparse
import csv
import importlib.util
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import requests
from bs4 import BeautifulSoup

# Try importing ddgs first (newer package), then duckduckgo_search
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


# Base directory is the project root (CENTROID/)
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = BASE_DIR / "outputs" / "word_freq_raw.csv"
DEFAULT_CORPUS_FILE = BASE_DIR / "outputs" / "corpus.txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("04_web_expand")


# Ensure project root is in sys.path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from pipeline.common import clean_html_to_text, normalize_unicode_punctuation
except ImportError:
    from common import clean_html_to_text, normalize_unicode_punctuation


def get_fetch_extractor() -> Optional[Callable[[str], str]]:
    """
    Attempts to locate and import the text extraction function from 01_fetch.py
    if present in the pipeline directory.
    """
    fetch_path = BASE_DIR / "pipeline" / "01_fetch.py"
    if not fetch_path.exists():
        return None

    try:
        spec = importlib.util.spec_from_file_location("pipeline_01_fetch", str(fetch_path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            func = getattr(mod, "extract_text_from_url", None)
            if callable(func):
                logger.info("Reusing extraction function 'extract_text_from_url' from 01_fetch.py")
                return func
    except Exception as e:
        logger.warning(f"Could not import extractor from {fetch_path}: {e}")

    return None


def fallback_extract_text_from_url(url: str, timeout: int = 15) -> str:
    """
    Fallback URL fetch and text extraction function when 01_fetch.py is
    not present or does not expose an extractor function.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "text" not in content_type and "html" not in content_type:
            logger.debug(f"Skipping non-HTML content-type '{content_type}' at {url}")
            return ""

        return clean_html_to_text(resp.text)
    except Exception as e:
        logger.warning(f"Failed to fetch/extract text from {url}: {e}")
        return ""


def read_top_words(csv_path: Path, top_n: int = 50) -> List[str]:
    """
    Reads outputs/word_freq_raw.csv and returns the top `top_n` words by count.

    Handles headers (e.g. 'word', 'count', 'source_file', 'tfidf_score') as well
    as headless 2-column CSVs, and aggregates counts if words appear across multiple rows.
    """
    if not csv_path.exists():
        logger.warning(f"Input file not found: {csv_path}")
        return []

    word_counts: Dict[str, int] = {}

    with open(csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue

            first_val = row[0].strip().lower()
            if first_val in ("word", "token", "term") and header is None:
                header = [c.strip().lower() for c in row]
                continue

            word = ""
            count = 1

            if header and "word" in header:
                word_idx = header.index("word")
                if len(row) > word_idx:
                    word = row[word_idx].strip()
                if "count" in header:
                    count_idx = header.index("count")
                    if len(row) > count_idx:
                        try:
                            count = int(float(row[count_idx].strip()))
                        except (ValueError, TypeError):
                            count = 1
            else:
                word = row[0].strip()
                if len(row) > 1:
                    try:
                        count = int(float(row[1].strip()))
                    except (ValueError, TypeError):
                        count = 1

            word_clean = word.lower().strip()
            # Basic validation: minimum length 3, alphabetic characters with optional hyphen/apostrophe
            if len(word_clean) >= 3 and re.match(r"^[a-z]+(?:[-'][a-z]+)*$", word_clean):
                word_counts[word_clean] = word_counts.get(word_clean, 0) + count

    # Sort descending by count
    sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    top_words = [w for w, _ in sorted_words[:top_n]]
    logger.info(f"Loaded top {len(top_words)} words from {csv_path}")
    return top_words


def search_duckduckgo(query: str, max_results: int = 3) -> List[str]:
    """
    Performs a DuckDuckGo web search for a given term and returns up to `max_results` URLs.
    """
    if DDGS is None:
        raise RuntimeError("duckduckgo_search / ddgs library is not installed. Please install it via requirements.txt.")

    urls: List[str] = []
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)
        if results:
            for item in results:
                url = item.get("href") or item.get("url") or item.get("link")
                if url and isinstance(url, str) and url.startswith("http"):
                    urls.append(url)
    except Exception as e:
        logger.warning(f"DuckDuckGo search error for query '{query}': {e}")

    return urls


def append_to_corpus(corpus_path: Path, query_word: str, url: str, text: str) -> None:
    """
    Appends extracted text to corpus.txt with a 'web-expand' source marker.
    """
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    marker = f"\n\n--- SOURCE: web-expand | query: {query_word} | url: {url} ---\n"
    with open(corpus_path, mode="a", encoding="utf-8") as f:
        f.write(marker)
        f.write(text.strip())
        f.write("\n")


def expand_corpus_from_web(
    top_n: int = 50,
    results_per_query: int = 3,
    rate_limit: float = 1.0,
    input_csv: Optional[Path] = None,
    output_corpus: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Main orchestration function for Stage 3 Web Expand:
      1. Reads top N words from word_freq_raw.csv.
      2. For each word, queries DuckDuckGo for top URLs with rate limiting.
      3. Fetches and extracts text using 01_fetch extractor (or fallback).
      4. Appends extracted text to corpus.txt with 'web-expand' marker.
      5. Does NOT run word scanning (reserved for Stage 2 / 03_scan_words.py).

    Returns a dictionary of execution metrics.
    """
    csv_file = input_csv or DEFAULT_INPUT_CSV
    corpus_file = output_corpus or DEFAULT_CORPUS_FILE

    top_words = read_top_words(csv_file, top_n=top_n)
    if not top_words:
        logger.warning("No words available to expand. Ensure outputs/word_freq_raw.csv exists and has data.")
        return {"words_queried": 0, "urls_fetched": 0, "bytes_appended": 0}

    # Resolve text extractor (reuse 01_fetch or use fallback)
    extractor = get_fetch_extractor() or fallback_extract_text_from_url

    visited_urls: Set[str] = set()
    total_urls_fetched = 0
    total_bytes_appended = 0

    logger.info(f"Starting web expansion for {len(top_words)} words (rate limit: {rate_limit}s/query)...")

    for idx, word in enumerate(top_words, start=1):
        logger.info(f"[{idx}/{len(top_words)}] Querying DuckDuckGo for word: '{word}'")
        search_urls = search_duckduckgo(word, max_results=results_per_query)

        for url in search_urls:
            if url in visited_urls:
                logger.debug(f"Skipping already visited URL: {url}")
                continue

            visited_urls.add(url)
            logger.info(f"  -> Fetching: {url}")
            extracted_text = extractor(url)

            if extracted_text and len(extracted_text.strip()) > 50:
                append_to_corpus(corpus_file, query_word=word, url=url, text=extracted_text)
                total_urls_fetched += 1
                total_bytes_appended += len(extracted_text.encode("utf-8"))
            else:
                logger.debug(f"  -> No meaningful text extracted from: {url}")

        # Rate-limiting sleep between DuckDuckGo search queries
        if idx < len(top_words) and rate_limit > 0:
            time.sleep(rate_limit)

    logger.info(
        f"Web expansion completed: {len(top_words)} words queried, "
        f"{total_urls_fetched} URLs appended to {corpus_file} ({total_bytes_appended:,} bytes)."
    )

    return {
        "words_queried": len(top_words),
        "urls_fetched": total_urls_fetched,
        "bytes_appended": total_bytes_appended,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CENTROID Pipeline Stage 3 - Web Expand: expands corpus using DuckDuckGo search queries on top words."
    )
    parser.add_argument(
        "--web-expand",
        action="store_true",
        help="Flag required to run web expansion. If not set, the stage is skipped."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Number of top words from raw frequency CSV to query (default: 50)."
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=3,
        help="Number of search result URLs to retrieve per word (default: 3)."
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds to sleep between search queries to avoid rate limits (default: 1.0)."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help=f"Path to input word frequency CSV (default: {DEFAULT_INPUT_CSV})."
    )
    parser.add_argument(
        "--output-corpus",
        type=str,
        default=None,
        help=f"Path to output corpus text file (default: {DEFAULT_CORPUS_FILE})."
    )

    args = parser.parse_args()

    if not args.web_expand:
        logger.info("--web-expand flag not specified. Skipping Stage 3 Web Expand.")
        sys.exit(0)

    input_csv = Path(args.input_csv) if args.input_csv else None
    output_corpus = Path(args.output_corpus) if args.output_corpus else None

    expand_corpus_from_web(
        top_n=args.top,
        results_per_query=args.results_per_query,
        rate_limit=args.rate_limit,
        input_csv=input_csv,
        output_corpus=output_corpus,
    )


if __name__ == "__main__":
    main()
