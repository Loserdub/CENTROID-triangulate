#!/usr/bin/env python3
"""
CENTROID Pipeline — Shared Core Utilities
Location: pipeline/common.py

Description:
  Consolidates common text normalization, wordlist loading, CSV serialization,
  and deterministic seeding utilities across CENTROID pipeline stages.
"""

import csv
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Try importing BeautifulSoup
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

logger = logging.getLogger("CENTROID_COMMON")


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Serializes an object to JSON, safely escaping '</' as '<\\/'
    to prevent closing HTML <script> tags unexpectedly or XSS injection.
    """
    return json.dumps(obj, **kwargs).replace("</", "<\\/")


def normalize_unicode_punctuation(text: str) -> str:
    """
    Normalizes typographic quotes, apostrophes, dashes, and ellipses into standard ASCII equivalents
    so that modern web articles and PDFs don't lose valid vocabulary words.
    """
    if not text:
        return ""
    # Typographic single quotes / apostrophes -> ASCII apostrophe
    text = re.sub(r"[\u2018\u2019\u201a\u201b\u2032]", "'", text)
    # Typographic double quotes -> ASCII double quote
    text = re.sub(r"[\u201c\u201d\u201e\u201f\u00ab\u00bb\u2033]", '"', text)
    # Typographic em-dash -> ASCII double hyphen
    text = re.sub(r"[\u2014\u2015]", "--", text)
    # Typographic dashes / hyphens -> ASCII hyphen
    text = re.sub(r"[\u2012\u2013\u2212]", "-", text)
    # Typographic ellipsis -> ASCII three dots
    text = re.sub(r"[\u2026]", "...", text)
    return text


def deterministic_word_seed(word: str) -> int:
    """
    Produces a completely deterministic 32-bit positive integer seed from any word string.
    Unlike Python's built-in hash() which is randomized per process by PYTHONHASHSEED,
    this MD5-derived seed is 100% reproducible across different Python processes,
    machines, and operating systems.
    """
    digest = hashlib.md5(word.encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def load_wordlist(file_path: Path) -> Set[str]:
    """
    Loads a wordlist from a text file (one word per line).
    Lowercases entries, strips whitespace, and skips empty lines or comments (#).
    """
    words: Set[str] = set()
    if not file_path.exists() or not file_path.is_file():
        return words

    try:
        with open(file_path, mode="r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                cleaned = line.strip().lower()
                if cleaned and not cleaned.startswith("#"):
                    words.add(cleaned)
    except Exception as e:
        logger.warning(f"Error reading wordlist '{file_path}': {e}")

    return words


def clean_html_to_text(html_content: str) -> str:
    """
    Parses HTML content and extracts clean textual content,
    removing scripts, styles, navigation, headers, footers, math typesetting,
    citation references, and web/wiki boilerplate.
    """
    if not html_content or not html_content.strip():
        return ""

    if BeautifulSoup is None:
        # Fallback regex-based HTML tag stripper if bs4 is unavailable
        stripped = re.sub(r"(?is)<script.*?</script>", " ", html_content)
        stripped = re.sub(r"(?is)<style.*?</style>", " ", stripped)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        return re.sub(r"\s+", " ", stripped).strip()

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content, multimedia, and layout tags
    for tag in soup([
        "script", "style", "nav", "footer", "header", "noscript",
        "svg", "form", "aside", "math", "annotation", "semantics",
        "audio", "video", "canvas", "figure", "figcaption", "iframe"
    ]):
        tag.decompose()

    # Remove common web, documentation, and Wikipedia boilerplate selectors
    boilerplate_selectors = [
        ".mw-editsection", ".reflist", ".references", "ol.references", "sup.reference",
        ".navbox", ".vertical-navbox", ".infobox", ".catlinks", "#catlinks", ".printfooter",
        ".noprint", ".mw-jump-link", ".citation", ".mw-cite-backlink", ".metadata",
        ".side-box", ".side-box-right", ".sister-bar", ".sistersitebox", ".sisterproject",
        ".portalbox", ".portal", ".ambox", ".cmbox", ".tmbox", ".fmbox", ".ombox",
        ".hatnote", ".shortdescription", ".mw-indicators", ".license-info", ".cc-license"
    ]
    for selector in boilerplate_selectors:
        for element in soup.select(selector):
            element.decompose()

    # Extract text with space separation
    text = soup.get_text(separator=" ", strip=True)

    # Strip licensing boilerplate and Commons cross-links
    text = re.sub(r"(?i)(?:wikimedia\s+commons|wikiquote|wikivoyage|wiktionary|wikisource|wikiversity|wikibooks)\s+has\s+.*?\.", " ", text)
    text = re.sub(r"(?i)creative\s+commons\s+attribution[^\.\n]*[\.\n]?", " ", text)
    text = re.sub(r"(?i)pages\s+displaying\s+short\s+descriptions\s+of\s+redirect\s+targets.*", " ", text)
    text = re.sub(r"(?i)text\s+is\s+available\s+under\s+the\s+creative\s+commons[^\.\n]*[\.\n]?", " ", text)

    # Strip raw embedded URLs
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def load_ranked_words_from_csv(ranked_csv_path: Path, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Safely loads ranked word frequencies from outputs/word_freq_ranked.csv.
    Expected columns: word, count, tfidf_score, combined_score, rank, cluster_id, cluster_label
    """
    if not ranked_csv_path.exists() or not ranked_csv_path.is_file():
        logger.warning(f"Ranked CSV file not found: {ranked_csv_path}")
        return []

    ranked_words: List[Dict[str, Any]] = []
    with open(ranked_csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or not row.get("word"):
                continue

            word = str(row["word"]).strip()
            if not word:
                continue

            try:
                rank = int(row.get("rank", len(ranked_words) + 1))
            except (ValueError, TypeError):
                rank = len(ranked_words) + 1

            try:
                count = int(float(row.get("count", 1)))
            except (ValueError, TypeError):
                count = 1

            try:
                tfidf = float(row.get("tfidf_score", 0.0))
            except (ValueError, TypeError):
                tfidf = 0.0

            try:
                combined = float(row.get("combined_score", 0.0))
            except (ValueError, TypeError):
                combined = 0.0

            cluster_id = str(row.get("cluster_id", "0")).strip()
            cluster_label = str(row.get("cluster_label", "cluster")).strip()

            ranked_words.append({
                "rank": rank,
                "word": word,
                "count": count,
                "tfidf_score": round(tfidf, 4),
                "combined_score": round(combined, 4),
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
            })

            if top_n is not None and len(ranked_words) >= top_n:
                break

    return ranked_words
