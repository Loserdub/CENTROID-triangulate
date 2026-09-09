#!/usr/bin/env python3
"""
CENTROID Pipeline — Stage 1: URL Fetch & Web Scraping
Specification: AGENT.md Section 4 Stage 1

Description:
  - Reads URLs from `inputs/urls.txt` (one URL per line).
  - Fetches each page via HTTP GET with robust headers and timeout.
  - Strips HTML tags, navigation, headers, footers, scripts, and styles using BeautifulSoup.
  - Flags low-content / near-empty / JS-boilerplate pages in `outputs/fetch_errors.log` as "low-content"
    instead of producing near-empty corpus entries.
  - Appends extracted clean text to `outputs/corpus.txt` with clear source boundary markers.
  - Skips and logs any unreachable or errored URLs without crashing.
  - Exports `extract_text_from_url` for reuse by `04_web_expand.py`.

Usage:
  python pipeline/01_fetch.py
  python pipeline/01_fetch.py --urls inputs/urls.txt --output outputs/corpus.txt
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup

# Base directory is the project root (CENTROID/)
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_URLS_FILE = BASE_DIR / "inputs" / "urls.txt"
DEFAULT_CORPUS_FILE = BASE_DIR / "outputs" / "corpus.txt"
DEFAULT_ERRORS_LOG = BASE_DIR / "outputs" / "fetch_errors.log"

# Ensure project root is in sys.path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from pipeline.common import clean_html_to_text as common_clean_html_to_text
except ImportError:
    from common import clean_html_to_text as common_clean_html_to_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("01_fetch")

# HTTP headers for polite web requests
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Safeguard thresholds
MIN_CONTENT_CHARS = 100
MIN_CONTENT_WORDS = 20
MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB per URL


def clean_html_to_text(html_content: str) -> str:
    """
    Parses HTML content and extracts meaningful textual content,
    removing scripts, styles, navigation, headers, footers, math typesetting,
    citation references, and web boilerplate.
    Preserves only the substantive article / report body.
    """
    return common_clean_html_to_text(html_content)


def check_is_low_content(text: str, min_chars: int = MIN_CONTENT_CHARS, min_words: int = MIN_CONTENT_WORDS) -> Tuple[bool, str]:
    """
    Checks if extracted plain text is near-empty or boilerplate (e.g. JS-rendered page with no SSR content).
    Returns (is_low, reason_description).
    """
    clean = text.strip()
    if not clean:
        return True, "Empty content after HTML cleaning"

    words = re.findall(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*", clean)
    if len(clean) < min_chars or len(words) < min_words:
        return True, f"Text below minimum threshold ({len(clean)} chars, {len(words)} words)"

    # Common boilerplate notices for non-rendered JavaScript single-page apps
    boilerplate_patterns = [
        r"you need to enable javascript to run this app",
        r"javascript is required to view this website",
        r"please enable javascript in your browser",
        r"enable javascript and refresh the page",
        r"loading\s*\.\.\.",
    ]
    lower_text = clean.lower()
    for pattern in boilerplate_patterns:
        if re.search(pattern, lower_text) and len(words) < 50:
            return True, f"Client-side JavaScript boilerplate notice detected ({pattern})"

    return False, ""


def log_fetch_error(log_path: Path, url: str, status: str, reason: str) -> None:
    """
    Logs an error / warning record to `outputs/fetch_errors.log`.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, mode="a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] URL: {url} | Status: {status} | Reason: {reason}\n")
    except Exception as e:
        logger.warning(f"Failed to write to '{log_path}': {e}")


def extract_text_from_url(
    url: str,
    timeout: int = 15,
    error_log_path: Optional[Path] = None,
    max_bytes: int = MAX_RESPONSE_BYTES
) -> str:
    """
    Fetches the given URL and extracts plain text content.
    Returns clean plain text, or empty string if failed / low-content.
    Hardened with Content-Type checks, size limits, and robust decoding.
    Flags low-content or failed responses in fetch_errors.log.
    Reusable by 04_web_expand.py.
    """
    clean_url = url.strip()
    if not clean_url:
        return ""

    log_file = error_log_path or DEFAULT_ERRORS_LOG

    try:
        # Stream response to validate headers before consuming full body
        with requests.get(clean_url, headers=DEFAULT_HEADERS, timeout=timeout, stream=True) as response:
            if response.status_code != 200:
                reason = f"HTTP {response.status_code}"
                logger.warning(f"HTTP error fetching URL '{clean_url}': {reason}")
                log_fetch_error(log_file, clean_url, f"http-error-{response.status_code}", reason)
                return ""

            # Content-Type check: only process HTML and text formats
            content_type = response.headers.get("Content-Type", "").lower()
            allowed_types = ("text/html", "text/plain", "application/xhtml+xml", "text/xml")
            if content_type and not any(ct in content_type for ct in allowed_types):
                reason = f"Unsupported Content-Type: '{content_type}'"
                logger.warning(f"Skipping URL '{clean_url}': {reason}")
                log_fetch_error(log_file, clean_url, "unsupported-content-type", reason)
                return ""

            # Check declared Content-Length if present
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        reason = f"Content-Length ({content_length} bytes) exceeds limit ({max_bytes} bytes)"
                        logger.warning(f"Skipping URL '{clean_url}': {reason}")
                        log_fetch_error(log_file, clean_url, "content-too-large", reason)
                        return ""
                except (ValueError, TypeError):
                    pass

            # Read content in chunks with byte-limit safety
            chunks = []
            bytes_read = 0
            for chunk in response.iter_content(chunk_size=65536, decode_unicode=False):
                chunks.append(chunk)
                bytes_read += len(chunk)
                if bytes_read > max_bytes:
                    reason = f"Body exceeded {max_bytes} bytes limit during streaming"
                    logger.warning(f"Aborting download for URL '{clean_url}': {reason}")
                    log_fetch_error(log_file, clean_url, "content-too-large", reason)
                    return ""

            raw_bytes = b"".join(chunks)
            encoding = response.encoding or response.apparent_encoding or "utf-8"
            try:
                html_text = raw_bytes.decode(encoding, errors="replace")
            except Exception:
                html_text = raw_bytes.decode("utf-8", errors="replace")

            text = clean_html_to_text(html_text)
            is_low, reason = check_is_low_content(text)
            if is_low:
                logger.warning(f"URL '{clean_url}' returned valid HTTP 200 but low-content: {reason}. Flagging in fetch_errors.log.")
                log_fetch_error(log_file, clean_url, "low-content", reason)
                return ""
            return text
    except Exception as e:
        reason = str(e)
        logger.warning(f"Failed to fetch URL '{clean_url}': {reason}")
        log_fetch_error(log_file, clean_url, "connection-error", reason)
        return ""


def load_urls(urls_path: Path) -> List[str]:
    """
    Loads target URLs from `inputs/urls.txt`.
    Ignores empty lines and comments (#).
    """
    if not urls_path.exists():
        logger.debug(f"URLs file not found: {urls_path}")
        return []

    urls: List[str] = []
    seen: Set[str] = set()
    try:
        with open(urls_path, mode="r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                cleaned = line.strip()
                if cleaned and not cleaned.startswith("#") and cleaned not in seen:
                    if cleaned.startswith("http://") or cleaned.startswith("https://"):
                        urls.append(cleaned)
                        seen.add(cleaned)
    except Exception as e:
        logger.warning(f"Error reading URLs file '{urls_path}': {e}")

    return urls


def fetch_urls_to_corpus(
    urls_path: Optional[Path] = None,
    output_corpus_path: Optional[Path] = None,
    error_log_path: Optional[Path] = None,
    timeout: int = 15,
    clean_corpus: bool = False
) -> Dict[str, int]:
    """
    Main orchestration function for Stage 1 URL fetching:
      - Reads URLs from `inputs/urls.txt`.
      - Extracts plain text for each URL.
      - Flags low-content pages in `outputs/fetch_errors.log` as "low-content".
      - Appends valid content to `outputs/corpus.txt` with boundary markers.
    """
    target_urls_file = urls_path or DEFAULT_URLS_FILE
    target_output_corpus = output_corpus_path or DEFAULT_CORPUS_FILE
    target_errors_log = error_log_path or DEFAULT_ERRORS_LOG

    if clean_corpus and target_output_corpus.exists():
        logger.info(f"Resetting existing corpus at '{target_output_corpus}' (--clean requested).")
        try:
            target_output_corpus.unlink()
        except Exception as e:
            logger.warning(f"Could not remove corpus file: {e}")

    urls = load_urls(target_urls_file)
    if not urls:
        logger.info(f"No URLs found in '{target_urls_file}'. Skipping URL fetch stage.")
        return {"urls_scanned": 0, "urls_succeeded": 0, "chars_extracted": 0}

    logger.info(f"Found {len(urls)} URL(s) to fetch from '{target_urls_file}'.")
    target_output_corpus.parent.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    total_chars = 0

    for idx, url in enumerate(urls, start=1):
        logger.info(f"[{idx}/{len(urls)}] Fetching: {url}")
        text = extract_text_from_url(url, timeout=timeout, error_log_path=target_errors_log)
        if text:
            char_count = len(text)
            total_chars += char_count
            succeeded += 1
            logger.info(f" -> Extracted {char_count:,} characters from {url}")

            # Append to corpus with standard boundary marker
            is_non_empty = target_output_corpus.exists() and target_output_corpus.stat().st_size > 0
            prefix = "\n\n" if is_non_empty else ""
            with open(target_output_corpus, mode="a", encoding="utf-8") as f:
                f.write(f"{prefix}--- SOURCE: url | url: {url} ---\n")
                f.write(text)
                f.write("\n")
        else:
            logger.warning(f" -> Skipped or empty content for: {url}")

    logger.info(f"URL fetch complete: {succeeded}/{len(urls)} URLs fetched ({total_chars:,} chars).")
    return {
        "urls_scanned": len(urls),
        "urls_succeeded": succeeded,
        "chars_extracted": total_chars
    }


def main():
    parser = argparse.ArgumentParser(
        description="CENTROID Pipeline Stage 1 — Fetch & Scrape URLs into outputs/corpus.txt"
    )
    parser.add_argument(
        "--urls",
        type=Path,
        default=DEFAULT_URLS_FILE,
        help=f"Path to urls.txt input file (default: {DEFAULT_URLS_FILE})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CORPUS_FILE,
        help=f"Path to output corpus file (default: {DEFAULT_CORPUS_FILE})"
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=DEFAULT_ERRORS_LOG,
        help=f"Path to error log file (default: {DEFAULT_ERRORS_LOG})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="HTTP request timeout in seconds (default: 15)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clear existing corpus file before fetching"
    )

    args = parser.parse_args()
    try:
        fetch_urls_to_corpus(
            urls_path=args.urls,
            output_corpus_path=args.output,
            error_log_path=args.error_log,
            timeout=args.timeout,
            clean_corpus=args.clean
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"URL fetch failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
