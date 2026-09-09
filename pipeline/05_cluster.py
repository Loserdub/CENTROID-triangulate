#!/usr/bin/env python3
"""
CENTROID Pipeline — Stage 4: Rank + Cluster
Specification: AGENT.md Section 4 Stage 4

Description:
  1. Reads `outputs/word_freq_raw.csv`.
  2. Computes `combined_score = (raw_count * 0.6) + (tfidf_score * 0.4)`.
  3. Sorts by combined score and selects the top N words (default: 1000).
  4. Builds a co-occurrence matrix using a sliding window of 5 words across `outputs/corpus.txt`.
  5. Trains a local Word2Vec model (gensim, 100 dimensions, window=5) on `outputs/corpus.txt`.
  6. Runs KMeans (scikit-learn, default k=10) on the word vectors for the top N words.
  7. Labels each cluster by its highest-frequency member word (anchor word).
  8. Outputs:
     - `outputs/word_freq_ranked.csv` (word, count, tfidf_score, combined_score, rank, cluster_id, cluster_label)
     - `outputs/clusters.json` ({cluster_id: {label, words: [...], total_count, pct_share}})

Usage:
  python pipeline/05_cluster.py
  python pipeline/05_cluster.py --top 500 --clusters 8
  python pipeline/05_cluster.py --raw-csv outputs/word_freq_raw.csv --corpus outputs/corpus.txt
"""

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

# Base directory is the project root (CENTROID/)
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_CSV = BASE_DIR / "outputs" / "word_freq_raw.csv"
DEFAULT_CORPUS_FILE = BASE_DIR / "outputs" / "corpus.txt"
DEFAULT_RANKED_CSV = BASE_DIR / "outputs" / "word_freq_ranked.csv"
DEFAULT_CLUSTERS_JSON = BASE_DIR / "outputs" / "clusters.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("05_cluster")

# Ensure project root is in sys.path for common imports
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from pipeline.common import deterministic_word_seed, normalize_unicode_punctuation
except ImportError:
    from common import deterministic_word_seed, normalize_unicode_punctuation


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes raw text into lowercase words adhering to CENTROID rules:
    - Lowercase tokens
    - Minimum length 3
    - Matching alphabetic characters with optional hyphens/apostrophes
    """
    cleaned = normalize_unicode_punctuation(text).lower()
    tokens = re.findall(r"[a-z]+(?:[-'][a-z]+)*", cleaned)
    return [t for t in tokens if len(t) >= 3]


def load_raw_frequencies(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Loads raw word frequencies from `outputs/word_freq_raw.csv`.
    Aggregates counts and tracks maximum/average TF-IDF scores if words appear across multiple files.

    Returns:
        List of dicts: [{'word': str, 'count': int, 'tfidf_score': float}, ...]
    """
    if not csv_path.exists():
        logger.warning(f"Raw word frequency file not found: {csv_path}")
        return []

    word_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "tfidf_score": 0.0, "tfidf_list": []})

    with open(csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue

            first_cell = row[0].strip().lower()
            if first_cell in ("word", "token", "term") and header is None:
                header = [c.strip().lower() for c in row]
                continue

            word = ""
            count = 1
            tfidf = 0.0

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
                if "tfidf_score" in header:
                    tfidf_idx = header.index("tfidf_score")
                    if len(row) > tfidf_idx:
                        try:
                            tfidf = float(row[tfidf_idx].strip())
                        except (ValueError, TypeError):
                            tfidf = 0.0
            else:
                word = row[0].strip()
                if len(row) > 1:
                    try:
                        count = int(float(row[1].strip()))
                    except (ValueError, TypeError):
                        count = 1
                if len(row) > 3:
                    try:
                        tfidf = float(row[3].strip())
                    except (ValueError, TypeError):
                        tfidf = 0.0

            clean_word = word.lower().strip()
            if clean_word and len(clean_word) >= 3 and re.match(r"^[a-z]+(?:[-'][a-z]+)*$", clean_word):
                word_data[clean_word]["count"] += count
                word_data[clean_word]["tfidf_list"].append(tfidf)

    results: List[Dict[str, Any]] = []
    for word, d in word_data.items():
        tfidf_scores = d["tfidf_list"]
        avg_tfidf = sum(tfidf_scores) / len(tfidf_scores) if tfidf_scores else 0.0
        results.append({
            "word": word,
            "count": d["count"],
            "tfidf_score": round(avg_tfidf, 4)
        })

    logger.info(f"Loaded {len(results):,} unique words from '{csv_path}'")
    return results


def rank_top_words(raw_words: List[Dict[str, Any]], top_n: int = 1000) -> List[Dict[str, Any]]:
    """
    Computes combined_score = (raw_count * 0.6) + (tfidf_score * 0.4) for each word,
    sorts descending by combined score, and selects the top N words.
    """
    for entry in raw_words:
        combined = (entry["count"] * 0.6) + (entry["tfidf_score"] * 0.4)
        entry["combined_score"] = round(combined, 4)

    # Sort descending by combined_score, then count, then alphabetical
    ranked = sorted(
        raw_words,
        key=lambda x: (x["combined_score"], x["count"], -ord(x["word"][0])),
        reverse=True
    )

    top_selected = ranked[:top_n]
    for idx, entry in enumerate(top_selected, start=1):
        entry["rank"] = idx

    logger.info(f"Ranked and selected top {len(top_selected):,} words (requested top {top_n}).")
    return top_selected


def load_and_tokenize_corpus(corpus_path: Path) -> List[List[str]]:
    """
    Loads `outputs/corpus.txt`, splits text into paragraphs/sentences,
    and returns a list of token lists for Word2Vec training and co-occurrence computation.
    """
    if not corpus_path.exists():
        logger.warning(f"Corpus file not found: {corpus_path}")
        return []

    try:
        with open(corpus_path, mode="r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        logger.warning(f"Failed to read corpus file '{corpus_path}': {e}")
        return []

    # Split corpus into sentences / logical chunks (paragraphs, newlines, sentence-enders)
    raw_sentences = re.split(r"(?:\r?\n\s*\r?\n|[\.\?!]\s+)", content)
    sentences: List[List[str]] = []

    for s in raw_sentences:
        # Ignore source boundary header lines
        if s.strip().startswith("--- SOURCE:"):
            continue
        tokens = tokenize_text(s)
        if len(tokens) >= 2:
            sentences.append(tokens)

    logger.info(f"Tokenized corpus into {len(sentences):,} sentences/chunks from '{corpus_path}'")
    return sentences


def build_cooccurrence_matrix(
    sentences: List[List[str]],
    top_words: List[str],
    window_size: int = 5
) -> Dict[str, Dict[str, int]]:
    """
    Builds a co-occurrence count matrix for the top N words using a sliding window of `window_size`.
    """
    top_word_set: Set[str] = set(top_words)
    cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    half_window = window_size // 2

    for sentence in sentences:
        s_len = len(sentence)
        for i, word in enumerate(sentence):
            if word in top_word_set:
                start = max(0, i - half_window)
                end = min(s_len, i + half_window + 1)
                for j in range(start, end):
                    if i != j:
                        context_word = sentence[j]
                        if context_word in top_word_set:
                            cooccurrence[word][context_word] += 1

    total_pairs = sum(len(ctx) for ctx in cooccurrence.values())
    logger.info(f"Built co-occurrence matrix: {len(cooccurrence):,} words tracked, {total_pairs:,} co-occurrence pairs (window={window_size}).")
    return cooccurrence


def train_word2vec_model(
    sentences: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    seed: int = 42
) -> Optional[Word2Vec]:
    """
    Trains a local Word2Vec model on the tokenized corpus.
    """
    if not sentences:
        logger.warning("No sentences available to train Word2Vec model.")
        return None

    logger.info(f"Training local Word2Vec model ({vector_size} dims, window={window}, seed={seed})...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=1,
        seed=seed,
        epochs=15
    )
    logger.info(f"Word2Vec training complete. Vocabulary size: {len(model.wv):,} words.")
    return model


def cluster_words(
    top_words: List[Dict[str, Any]],
    w2v_model: Optional[Word2Vec],
    n_clusters: int = 10,
    random_state: int = 42
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Clusters the top N words using KMeans on their 100-dimensional Word2Vec vectors.
    Labels each cluster by its highest-frequency member word.

    Returns:
        (ranked_words_with_clusters, clusters_summary_dict)
    """
    if not top_words:
        return [], {}

    if len(top_words) < n_clusters:
        actual_k = max(1, len(top_words))
        warn_msg = (
            f"Requested {n_clusters} clusters, but only {len(top_words)} unique word(s) exist. "
            f"Auto-reducing cluster count to {actual_k} to prevent clustering error."
        )
        logger.warning(warn_msg)
        print(f"\n[WARNING] {warn_msg}\n")
    else:
        actual_k = max(1, n_clusters)

    vector_size = 100
    vectors: List[np.ndarray] = []
    vector_dim = w2v_model.vector_size if w2v_model else vector_size

    # Extract or generate vector for each top word
    for entry in top_words:
        word = entry["word"]
        if w2v_model and word in w2v_model.wv:
            vec = w2v_model.wv[word]
        else:
            # Fallback deterministic pseudo-vector if word wasn't in corpus
            word_seed = deterministic_word_seed(word)
            rng = np.random.RandomState(word_seed)
            vec = rng.normal(loc=0.0, scale=0.1, size=vector_dim).astype(np.float32)
        vectors.append(vec)

    X = np.array(vectors)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    # Perform KMeans clustering
    logger.info(f"Running KMeans clustering with k={actual_k} on {len(top_words):,} word vectors...")
    kmeans = KMeans(n_clusters=actual_k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # Group words by cluster ID
    clusters_grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for entry, cid in zip(top_words, cluster_labels):
        entry["cluster_id"] = int(cid)
        clusters_grouped[int(cid)].append(entry)

    # Ensure combined_score is present on each entry
    for entry in top_words:
        if "combined_score" not in entry:
            entry["combined_score"] = round(entry.get("count", 0) * 0.6 + entry.get("tfidf_score", 0.0) * 0.4, 4)

    # Identify anchor word (highest count) for each cluster
    cluster_anchor_labels: Dict[int, str] = {}
    cluster_totals: Dict[int, int] = {}
    total_top_words_count = sum(entry.get("count", 0) for entry in top_words)

    for cid, members in clusters_grouped.items():
        # Anchor is member with highest raw count, tie-breaker combined_score
        members_sorted = sorted(members, key=lambda m: (m.get("count", 0), m.get("combined_score", 0.0)), reverse=True)
        anchor_word = members_sorted[0]["word"]
        cluster_anchor_labels[cid] = anchor_word
        cluster_totals[cid] = sum(m.get("count", 0) for m in members)

    # Assign cluster_label to each top word entry
    for entry in top_words:
        cid = entry["cluster_id"]
        entry["cluster_label"] = cluster_anchor_labels[cid]

    # Build outputs/clusters.json structure
    clusters_output: Dict[str, Any] = {}
    
    # Sort cluster IDs by total count descending
    sorted_cluster_ids = sorted(clusters_grouped.keys(), key=lambda c: cluster_totals[c], reverse=True)

    for new_idx, cid in enumerate(sorted_cluster_ids):
        members = clusters_grouped[cid]
        members_sorted = sorted(members, key=lambda m: (m.get("count", 0), m.get("combined_score", 0.0)), reverse=True)
        member_words = [m["word"] for m in members_sorted]
        c_total = cluster_totals[cid]
        pct_share = round((c_total / total_top_words_count * 100), 2) if total_top_words_count > 0 else 0.0

        clusters_output[str(cid)] = {
            "label": cluster_anchor_labels[cid],
            "words": member_words,
            "total_count": c_total,
            "pct_share": pct_share
        }

    return top_words, clusters_output


def save_ranked_csv(ranked_words: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Saves ranked words to `outputs/word_freq_ranked.csv`.
    Columns: word, count, tfidf_score, combined_score, rank, cluster_id, cluster_label
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["word", "count", "tfidf_score", "combined_score", "rank", "cluster_id", "cluster_label"]

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in ranked_words:
            writer.writerow({
                "word": entry["word"],
                "count": entry["count"],
                "tfidf_score": entry["tfidf_score"],
                "combined_score": entry["combined_score"],
                "rank": entry["rank"],
                "cluster_id": entry["cluster_id"],
                "cluster_label": entry["cluster_label"]
            })
    logger.info(f"Saved ranked word frequencies to '{output_path}' ({len(ranked_words):,} rows)")


def save_clusters_json(clusters_data: Dict[str, Any], output_path: Path) -> None:
    """
    Saves cluster summary dictionary to `outputs/clusters.json`.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(clusters_data, f, indent=2)
    logger.info(f"Saved cluster definitions to '{output_path}' ({len(clusters_data)} clusters)")


def run_clustering_pipeline(
    top_n: int = 1000,
    n_clusters: int = 10,
    raw_csv_path: Optional[Path] = None,
    corpus_path: Optional[Path] = None,
    output_ranked_path: Optional[Path] = None,
    output_clusters_path: Optional[Path] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Orchestrates Stage 4 clustering pipeline.
    """
    raw_csv = raw_csv_path or DEFAULT_RAW_CSV
    corpus_file = corpus_path or DEFAULT_CORPUS_FILE
    ranked_csv = output_ranked_path or DEFAULT_RANKED_CSV
    clusters_json = output_clusters_path or DEFAULT_CLUSTERS_JSON

    # 1. Read outputs/word_freq_raw.csv
    raw_words = load_raw_frequencies(raw_csv)
    if not raw_words:
        logger.warning("No words found in raw frequency CSV. Creating empty output files.")
        save_ranked_csv([], ranked_csv)
        save_clusters_json({}, clusters_json)
        return {"words_ranked": 0, "clusters_created": 0}

    # 2. Compute combined scores & rank top N words
    top_words = rank_top_words(raw_words, top_n=top_n)
    top_word_list = [entry["word"] for entry in top_words]

    # 3. Load & tokenize outputs/corpus.txt
    sentences = load_and_tokenize_corpus(corpus_file)

    # 4. Build co-occurrence matrix (sliding window of 5 words)
    build_cooccurrence_matrix(sentences, top_word_list, window_size=5)

    # 5. Train local Word2Vec model (100 dimensions)
    w2v_model = train_word2vec_model(sentences, vector_size=100, window=5, seed=seed)

    # 6. Run KMeans clustering
    ranked_words_with_clusters, clusters_summary = cluster_words(
        top_words=top_words,
        w2v_model=w2v_model,
        n_clusters=n_clusters,
        random_state=seed
    )

    # 7. Output files
    save_ranked_csv(ranked_words_with_clusters, ranked_csv)
    save_clusters_json(clusters_summary, clusters_json)

    # Print summary table to console
    logger.info("=" * 60)
    logger.info(f"CENTROID CLUSTERING SUMMARY (Top {len(ranked_words_with_clusters)} Words | {len(clusters_summary)} Clusters)")
    logger.info("=" * 60)
    for cid, data in clusters_summary.items():
        sample_words = ", ".join(data["words"][:6])
        if len(data["words"]) > 6:
            sample_words += f", ... (+{len(data['words']) - 6} more)"
        logger.info(f"Cluster {cid} [\"{data['label']}\"]: {data['total_count']:,} tokens ({data['pct_share']}%) | Words: {sample_words}")
    logger.info("=" * 60)

    return {
        "words_ranked": len(ranked_words_with_clusters),
        "clusters_created": len(clusters_summary)
    }


def main():
    parser = argparse.ArgumentParser(
        description="CENTROID Pipeline Stage 4 — Rank + Cluster words using Word2Vec and KMeans."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=1000,
        help="Number of top words to rank and cluster (default: 1000)."
    )
    parser.add_argument(
        "--clusters", "-k",
        type=int,
        default=10,
        help="Number of KMeans clusters (default: 10)."
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=DEFAULT_RAW_CSV,
        help=f"Path to input raw word frequency CSV (default: {DEFAULT_RAW_CSV})."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_FILE,
        help=f"Path to input corpus text file (default: {DEFAULT_CORPUS_FILE})."
    )
    parser.add_argument(
        "--output-ranked",
        type=Path,
        default=DEFAULT_RANKED_CSV,
        help=f"Path to output ranked CSV (default: {DEFAULT_RANKED_CSV})."
    )
    parser.add_argument(
        "--output-clusters",
        type=Path,
        default=DEFAULT_CLUSTERS_JSON,
        help=f"Path to output clusters JSON (default: {DEFAULT_CLUSTERS_JSON})."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible Word2Vec and KMeans (default: 42)."
    )

    args = parser.parse_args()
    try:
        run_clustering_pipeline(
            top_n=args.top,
            n_clusters=args.clusters,
            raw_csv_path=args.raw_csv,
            corpus_path=args.corpus,
            output_ranked_path=args.output_ranked,
            output_clusters_path=args.output_clusters,
            seed=args.seed
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Clustering execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
