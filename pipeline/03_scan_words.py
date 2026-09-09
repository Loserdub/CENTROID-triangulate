#!/usr/bin/env python3
"""
CENTROID Pipeline — Stage 2: Word Scanner
Specification: AGENT.md Section 5 (Word Scanner Rules) & Section 4 Stage 2

Description:
  Reads `outputs/corpus.txt`, segments it by source document boundaries,
  and applies Word Scanner Rules 1 through 9:
    RULE 1: Lowercase everything
    RULE 2: Remove tokens < 3 characters
    RULE 3: Remove tokens matching [^a-z\\-'] (numbers, symbols)
    RULE 4: Strip stopwords (EN + custom list)
    RULE 5: If target_words.txt exists, flag matches before any filtering
    RULE 6: Optional stemming (off by default; enable via --stem flag)
    RULE 7: Count frequency per source file AND globally
    RULE 8: Compute TF-IDF across all source documents
    RULE 9: Export raw CSV with all surviving tokens

Output:
  outputs/word_freq_raw.csv (columns: word, count, source_file, tfidf_score)
  Console sanity check printing top 20 words by raw count.

Usage:
  python pipeline/03_scan_words.py
  python pipeline/03_scan_words.py --stem
  python pipeline/03_scan_words.py --corpus outputs/corpus.txt --output outputs/word_freq_raw.csv
"""

import argparse
import csv
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Try importing NLTK components
try:
    import nltk
    from nltk.stem.porter import PorterStemmer
except ImportError:
    nltk = None
    PorterStemmer = None


# Base directory is the project root (CENTROID/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Ensure project root is in sys.path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from pipeline.common import load_wordlist as common_load_wordlist, normalize_unicode_punctuation as common_normalize_unicode_punctuation
except ImportError:
    from common import load_wordlist as common_load_wordlist, normalize_unicode_punctuation as common_normalize_unicode_punctuation

# Default file paths
DEFAULT_CORPUS_FILE = BASE_DIR / "outputs" / "corpus.txt"
DEFAULT_OUTPUT_CSV = BASE_DIR / "outputs" / "word_freq_raw.csv"
DEFAULT_STOPWORDS_FILE = BASE_DIR / "wordlists" / "stopwords_en.txt"
DEFAULT_CUSTOM_IGNORE_FILE = BASE_DIR / "wordlists" / "custom_ignore.txt"
DEFAULT_WEB_LINGO_FILE = BASE_DIR / "wordlists" / "web_coding_lingo.txt"
DEFAULT_TARGET_WORDS_FILE = BASE_DIR / "wordlists" / "target_words.txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("03_scan_words")


def load_wordlist(file_path: Path) -> Set[str]:
    """
    Loads a wordlist from a text file (one word per line).
    Lowercases entries, strips whitespace, and skips empty lines or comments (#).
    """
    return common_load_wordlist(file_path)


def normalize_unicode_punctuation(text: str) -> str:
    """
    Normalizes typographic quotes, apostrophes, and dashes into standard ASCII equivalents.
    """
    return common_normalize_unicode_punctuation(text)


def parse_corpus_documents(corpus_path: Path) -> List[Tuple[str, str]]:
    """
    Reads outputs/corpus.txt and splits it into individual source documents.
    Recognizes boundary markers such as:
      --- SOURCE: pdf | file: filename.pdf ---
      --- SOURCE: docx | file: filename.docx ---
      --- SOURCE: txt | file: filename.txt ---
      --- SOURCE: url | url: https://... ---
      --- SOURCE: web-expand | query: word | url: https://... ---
      === SOURCE: ... ===

    Returns a list of (source_identifier, document_text) tuples.
    If multiple chunks belong to the same source, their texts are merged.
    If no source markers are present, treats the entire file as a single document.
    """
    if not corpus_path.exists():
        logger.warning(f"Corpus file does not exist: {corpus_path}")
        return []

    try:
        with open(corpus_path, mode="r", encoding="utf-8", errors="replace") as f:
            corpus_text = f.read()
    except Exception as e:
        logger.error(f"Failed to read corpus file '{corpus_path}': {e}")
        return []

    if not corpus_text.strip():
        logger.warning(f"Corpus file is empty: {corpus_path}")
        return []

    # Regex matching boundary markers: --- SOURCE: ... --- or === SOURCE: ... ===
    marker_pattern = re.compile(r"(?m)^[-=]{3,}\s*SOURCE:\s*(.*?)\s*[-=]{3,}\s*$")
    matches = list(marker_pattern.finditer(corpus_text))

    raw_docs: List[Tuple[str, str]] = []

    if not matches:
        # No source boundary markers found; treat whole corpus as single document
        raw_docs.append(("corpus.txt", corpus_text.strip()))
    else:
        # Check if there is text before the first marker
        pre_text = corpus_text[:matches[0].start()].strip()
        if pre_text:
            raw_docs.append(("corpus.txt", pre_text))

        for i, match in enumerate(matches):
            header = match.group(1).strip()
            start_pos = match.end()
            end_pos = matches[i + 1].start() if (i + 1) < len(matches) else len(corpus_text)
            doc_text = corpus_text[start_pos:end_pos].strip()

            # Extract human-readable source identifier from header
            file_match = re.search(r"file:\s*([^\s|]+)", header, re.IGNORECASE)
            url_match = re.search(r"url:\s*([^\s|]+)", header, re.IGNORECASE)

            if file_match:
                source_id = file_match.group(1).strip()
            elif url_match:
                source_id = url_match.group(1).strip()
            else:
                source_id = header if header else f"doc_{i + 1}"

            if doc_text:
                raw_docs.append((source_id, doc_text))

    # Merge multiple text blocks that share the exact same source identifier
    merged_docs: Dict[str, List[str]] = defaultdict(list)
    for src_id, text in raw_docs:
        merged_docs[src_id].append(text)

    final_docs: List[Tuple[str, str]] = [
        (src_id, "\n\n".join(texts)) for src_id, texts in merged_docs.items()
    ]

    return final_docs


def normalize_unicode_punctuation(text: str) -> str:
    """
    Normalizes typographic quotes, apostrophes, and dashes into standard ASCII equivalents
    so that modern web articles and PDFs don't lose valid vocabulary words.
    """
    # Typographic single quotes / apostrophes -> ASCII apostrophe
    text = re.sub(r"[\u2018\u2019\u201a\u201b\u2032]", "'", text)
    # Typographic double quotes -> ASCII double quote
    text = re.sub(r"[\u201c\u201d\u201e\u201f\u00ab\u00bb\u2033]", '"', text)
    # Typographic dashes / hyphens -> ASCII hyphen
    text = re.sub(r"[\u2012\u2013\u2014\u2015\u2212]", "-", text)
    return text


def tokenize_raw_text(text: str) -> List[str]:
    """
    Tokenizes raw text using NLTK word_tokenize if available,
    with a robust fallback to regex-based word boundary tokenization.
    """
    norm_text = normalize_unicode_punctuation(text)
    if nltk is not None:
        try:
            return nltk.word_tokenize(norm_text)
        except LookupError:
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                return nltk.word_tokenize(norm_text)
            except Exception:
                pass
        except Exception:
            pass

    # Regex tokenization fallback: matches alphabetic/alphanumeric words with internal hyphens or apostrophes
    return re.findall(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*", norm_text)


def process_document_tokens(
    text: str,
    stopwords: Set[str],
    target_words: Set[str],
    stem: bool = False,
    stemmer: Optional[Any] = None,
) -> List[str]:
    """
    Processes and filters tokens for a document according to Word Scanner Rules:
      RULE 1: Lowercase everything
      RULE 5: If target_words.txt exists, flag matches before any filtering
      RULE 2: Remove tokens < 3 characters
      RULE 3: Remove tokens matching [^a-z\\-'] (numbers, symbols)
      RULE 4: Strip stopwords (EN + custom list)
      RULE 6: Optional stemming (via --stem flag using Porter Stemmer)

    Returns:
      List of surviving tokens for the document.
    """
    # RULE 1: Lowercase everything and normalize unicode typography
    lowercased_text = normalize_unicode_punctuation(text).lower()
    raw_tokens = tokenize_raw_text(lowercased_text)

    # Valid word pattern: alphabetic characters with optional internal hyphens or apostrophes
    valid_word_pattern = re.compile(r"^[a-z]+(?:[-'][a-z]+)*$")

    surviving_tokens: List[str] = []

    for raw_tok in raw_tokens:
        # Strip surrounding punctuation / whitespace
        tok = raw_tok.strip(" \t\n\r\"'.,;:!?()[]{}<>~`_/\\“”‘’—–")
        if not tok:
            continue

        # RULE 5: Check against target_words.txt before any filtering
        is_target_word = tok in target_words

        if not is_target_word:
            # RULE 2: Remove tokens < 3 characters
            if len(tok) < 3:
                continue

            # RULE 3: Remove tokens matching [^a-z\-'] (numbers, symbols, punctuation)
            if not valid_word_pattern.match(tok):
                continue

            # RULE 4: Strip stopwords (EN + custom list)
            if tok in stopwords:
                continue

        # RULE 6: Optional stemming (Porter Stemmer)
        if stem and stemmer is not None:
            stemmed = stemmer.stem(tok)
            if stemmed:
                tok = stemmed

        surviving_tokens.append(tok)

    return surviving_tokens


def compute_tfidf_scores(
    doc_token_counts: Dict[str, Counter],
) -> Dict[str, Dict[str, float]]:
    """
    RULE 8: Computes TF-IDF per word across all source documents.
    Treats each original fetched/extracted document as one document.

    Uses smooth inverse document frequency:
      IDF(w) = ln((1 + N) / (1 + DF(w))) + 1.0
      TF-IDF_raw(w, d) = count(w, d) * IDF(w)
      TF-IDF(w, d) = TF-IDF_raw(w, d) / L2_norm(d)

    Returns:
      Dict mapping doc_id -> {word: tfidf_score}
    """
    num_docs = len(doc_token_counts)
    if num_docs == 0:
        return {}

    # Calculate document frequency DF(w) for each word
    doc_freqs: Counter = Counter()
    for doc_id, counts in doc_token_counts.items():
        for word in counts:
            doc_freqs[word] += 1

    # Compute Smooth IDF for each word
    idfs: Dict[str, float] = {}
    for word, df in doc_freqs.items():
        idfs[word] = math.log((1 + num_docs) / (1 + df)) + 1.0

    # Compute L2-normalized TF-IDF per document
    tfidf_by_doc: Dict[str, Dict[str, float]] = {}

    for doc_id, counts in doc_token_counts.items():
        raw_tfidf = {w: count * idfs[w] for w, count in counts.items()}
        l2_norm = math.sqrt(sum(val ** 2 for val in raw_tfidf.values()))

        tfidf_by_doc[doc_id] = {}
        for w, raw_val in raw_tfidf.items():
            score = (raw_val / l2_norm) if l2_norm > 0.0 else 0.0
            tfidf_by_doc[doc_id][w] = round(score, 4)

    return tfidf_by_doc


def scan_words(
    corpus_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
    stopwords_path: Optional[Path] = None,
    custom_ignore_path: Optional[Path] = None,
    web_lingo_path: Optional[Path] = None,
    target_words_path: Optional[Path] = None,
    stem: bool = False,
    top_sanity_check: int = 20,
    min_words: int = 50,
) -> Dict[str, Any]:
    """
    Main orchestration function for Stage 2 Word Scanner:
      1. Loads stopwords, custom ignore words, web/coding lingo, and target words.
      2. Reads and parses outputs/corpus.txt into source documents.
      3. Applies Rules 1 through 6 to extract surviving tokens per document.
      4. Computes per-source and global frequency counts (RULE 7).
      5. Computes TF-IDF scores across all source documents (RULE 8).
      6. Exports outputs/word_freq_raw.csv with columns: word, count, source_file, tfidf_score (RULE 9).
      7. Prints top N words by raw count to console as a sanity check.

    Returns:
      Dictionary containing execution statistics and metrics.
    """
    target_corpus = corpus_path or DEFAULT_CORPUS_FILE
    target_output_csv = output_csv_path or DEFAULT_OUTPUT_CSV
    target_stopwords = stopwords_path or DEFAULT_STOPWORDS_FILE
    target_custom_ignore = custom_ignore_path or DEFAULT_CUSTOM_IGNORE_FILE
    target_web_lingo = web_lingo_path or DEFAULT_WEB_LINGO_FILE
    target_target_words = target_words_path or DEFAULT_TARGET_WORDS_FILE

    # Ensure output directory exists
    target_output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load wordlists
    stopwords_en = load_wordlist(target_stopwords)
    custom_ignore = load_wordlist(target_custom_ignore)
    web_lingo = load_wordlist(target_web_lingo)
    combined_stopwords = stopwords_en | custom_ignore | web_lingo
    target_words = load_wordlist(target_target_words)

    logger.info(
        f"Loaded wordlists: {len(stopwords_en)} EN stopwords, {len(custom_ignore)} custom ignore words, "
        f"{len(web_lingo)} web/coding lingo words ({len(combined_stopwords)} total excluded words), "
        f"{len(target_words)} target words."
    )

    # Initialize Porter Stemmer if stem flag is requested
    stemmer = None
    if stem:
        if PorterStemmer is not None:
            stemmer = PorterStemmer()
            logger.info("Porter Stemmer enabled (--stem).")
        else:
            logger.warning("NLTK PorterStemmer not available; proceeding without stemming.")

    # 2. Parse corpus into source documents
    documents = parse_corpus_documents(target_corpus)
    total_raw_words = sum(len(doc_text.split()) for _, doc_text in documents) if documents else 0

    if not documents or total_raw_words < min_words:
        msg = f"Corpus at '{target_corpus}' is empty or contains below {min_words} words ({total_raw_words} words found). Minimum {min_words} words required for word scanning."
        logger.warning(msg)
        print(f"\n[CENTROID Word Scanner] Notice: {msg}\nExiting without generating zero-row CSV.\n")
        return {
            "documents_count": len(documents),
            "total_tokens": 0,
            "unique_words": 0,
            "top_words": [],
            "status": "insufficient_corpus",
        }

    logger.info(f"Parsed {len(documents)} source document(s) ({total_raw_words:,} raw words) from '{target_corpus}'.")

    # 3. Process tokens per document (RULE 1 - RULE 6) & Count frequencies (RULE 7)
    doc_token_counts: Dict[str, Counter] = {}
    global_counts: Counter = Counter()
    total_token_occurrences = 0

    for source_id, doc_text in documents:
        tokens = process_document_tokens(
            text=doc_text,
            stopwords=combined_stopwords,
            target_words=target_words,
            stem=stem,
            stemmer=stemmer,
        )
        counts = Counter(tokens)
        doc_token_counts[source_id] = counts
        global_counts.update(counts)
        total_token_occurrences += len(tokens)

    # 4. Compute TF-IDF across source documents (RULE 8)
    tfidf_by_doc = compute_tfidf_scores(doc_token_counts)

    # 5. Prepare rows for raw CSV export (RULE 9)
    # Output columns: word, count, source_file, tfidf_score
    csv_rows: List[Tuple[str, int, str, float]] = []
    for source_id, counts in doc_token_counts.items():
        for word, count in counts.items():
            tfidf_score = tfidf_by_doc.get(source_id, {}).get(word, 0.0)
            csv_rows.append((word, count, source_id, tfidf_score))

    # Sort rows by count descending, then tfidf_score descending, then word
    csv_rows.sort(key=lambda item: (-item[1], -item[3], item[0]))

    # Write outputs/word_freq_raw.csv
    with open(target_output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "count", "source_file", "tfidf_score"])
        for word, count, source_file, tfidf in csv_rows:
            writer.writerow([word, count, source_file, f"{tfidf:.4f}"])

    logger.info(
        f"Exported {len(csv_rows)} records ({len(global_counts)} unique words) to '{target_output_csv}'."
    )

    # 6. Sanity check: Print top N words by raw count to console
    top_words = global_counts.most_common(top_sanity_check)

    # Calculate document appearance count per top word
    doc_freqs: Counter = Counter()
    for counts in doc_token_counts.values():
        for word in counts:
            doc_freqs[word] += 1

    print("\n" + "=" * 64)
    print(f" CENTROID Word Scanner - Top {len(top_words)} Words (Sanity Check)")
    print("=" * 64)
    print(f" {'Rank':<5} {'Word':<24} {'Raw Count':<12} {'Doc Count':<10} {'Target?':<8}")
    print("-" * 64)
    for rank, (word, raw_cnt) in enumerate(top_words, start=1):
        is_tgt = "YES" if word in target_words else "-"
        df_cnt = doc_freqs.get(word, 0)
        print(f" {rank:<5} {word:<24} {raw_cnt:<12} {df_cnt:<10} {is_tgt:<8}")
    print("=" * 64)
    print(f" Total Documents:  {len(documents)}")
    print(f" Total Tokens:     {total_token_occurrences:,}")
    print(f" Unique Words:     {len(global_counts):,}")
    print(f" Stemming:         {'Enabled' if stem else 'Disabled'}")
    print(f" Output Written:   {target_output_csv}")
    print("=" * 64 + "\n")

    return {
        "documents_count": len(documents),
        "total_tokens": total_token_occurrences,
        "unique_words": len(global_counts),
        "top_words": top_words,
        "output_file": str(target_output_csv),
    }


def main():
    parser = argparse.ArgumentParser(
        description="CENTROID Pipeline Stage 2 — Word Scanner: Tokenize, filter, count, and compute TF-IDF."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_FILE,
        help=f"Path to input corpus file (default: {DEFAULT_CORPUS_FILE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Path to output CSV file (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--stopwords",
        type=Path,
        default=DEFAULT_STOPWORDS_FILE,
        help=f"Path to English stopwords file (default: {DEFAULT_STOPWORDS_FILE})",
    )
    parser.add_argument(
        "--custom-ignore",
        type=Path,
        default=DEFAULT_CUSTOM_IGNORE_FILE,
        help=f"Path to custom ignore list (default: {DEFAULT_CUSTOM_IGNORE_FILE})",
    )
    parser.add_argument(
        "--web-lingo",
        type=Path,
        default=DEFAULT_WEB_LINGO_FILE,
        help=f"Path to web and coding lingo exclusion list (default: {DEFAULT_WEB_LINGO_FILE})",
    )
    parser.add_argument(
        "--target-words",
        type=Path,
        default=DEFAULT_TARGET_WORDS_FILE,
        help=f"Path to target words list (default: {DEFAULT_TARGET_WORDS_FILE})",
    )
    parser.add_argument(
        "--stem",
        action="store_true",
        default=False,
        help="Enable Porter Stemmer (default: off)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top words to print in sanity check (default: 20)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=50,
        help="Minimum words required in corpus to proceed with scanning (default: 50)",
    )

    args = parser.parse_args()

    try:
        metrics = scan_words(
            corpus_path=args.corpus,
            output_csv_path=args.output,
            stopwords_path=args.stopwords,
            custom_ignore_path=args.custom_ignore,
            web_lingo_path=args.web_lingo,
            target_words_path=args.target_words,
            stem=args.stem,
            top_sanity_check=args.top,
            min_words=args.min_words,
        )
        if metrics.get("status") == "insufficient_corpus":
            sys.exit(1)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error during word scanning: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
