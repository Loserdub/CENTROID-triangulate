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
import datetime
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# Base directory is the project root (CENTROID/)
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_CSV = BASE_DIR / "outputs" / "word_freq_raw.csv"
DEFAULT_CORPUS_FILE = BASE_DIR / "outputs" / "corpus.txt"
DEFAULT_RANKED_CSV = BASE_DIR / "outputs" / "word_freq_ranked.csv"
DEFAULT_CLUSTERS_JSON = BASE_DIR / "outputs" / "clusters.json"
DEFAULT_MEDOIDS_JSON = BASE_DIR / "outputs" / "medoids.json"
DEFAULT_TARGET_WORDS_FILE = BASE_DIR / "wordlists" / "target_words.txt"
DEFAULT_VECTORS_NPY = BASE_DIR / "outputs" / "vectors.npy"
DEFAULT_VOCAB_JSON = BASE_DIR / "outputs" / "vocab.json"

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
    from pipeline.common import atomic_write_json, deterministic_word_seed, normalize_unicode_punctuation
    from pipeline.graph_community import run_cooccurrence_graph_pipeline
except ImportError:
    from common import atomic_write_json, deterministic_word_seed, normalize_unicode_punctuation
    from graph_community import run_cooccurrence_graph_pipeline


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


def compute_cluster_medoid_pairwise(
    cluster_vectors: np.ndarray,
    metric: str = "cosine"
) -> Tuple[int, float, float]:
    """
    Plan B Intra-Cluster Medoid Discovery (BLAS Accelerated):
    Computes empirical point minimizing the sum of distances to other members.
    For cosine distance, uses vectorized matrix dot-product kernel O(m*d)
    avoiding O(m^2) distance matrix allocations.

    Returns:
        (medoid_idx, average_dispersion_radius, max_distance)
    """
    n_samples = len(cluster_vectors)
    if n_samples == 0:
        return 0, 0.0, 0.0
    if n_samples == 1:
        return 0, 0.0, 0.0

    if metric == "cosine":
        # BLAS-accelerated kernel via unit-normalized dot products
        norms = np.linalg.norm(cluster_vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        X_norm = cluster_vectors / norms

        # S_sum is the sum of all normalized vectors in the cluster: shape (D,)
        S_sum = np.sum(X_norm, axis=0)

        # sum_j (1 - X_i . X_j) = n_samples - (X_i . S_sum)
        # Minimizing distance sum is mathematically equivalent to maximizing dot product with S_sum
        total_sim = np.dot(X_norm, S_sum)
        medoid_idx = int(np.argmax(total_sim))

        # Vectorized distances to chosen medoid: 1.0 - (X_norm . X_norm[medoid_idx])
        dists_to_medoid = 1.0 - np.dot(X_norm, X_norm[medoid_idx])
        dists_to_medoid = np.clip(dists_to_medoid, 0.0, 2.0)
        dispersion_radius = float(np.mean(dists_to_medoid))
        max_radius = float(np.max(dists_to_medoid))

        return medoid_idx, dispersion_radius, max_radius
    else:
        # Fallback for non-cosine metrics
        dist_matrix = cdist(cluster_vectors, cluster_vectors, metric=metric)
        dist_matrix = np.nan_to_num(dist_matrix, nan=0.0, posinf=1.0, neginf=0.0)
        total_distances = dist_matrix.sum(axis=1)
        medoid_idx = int(np.argmin(total_distances))
        dispersion_radius = float(np.mean(dist_matrix[medoid_idx]))
        max_radius = float(np.max(dist_matrix[medoid_idx]))
        return medoid_idx, dispersion_radius, max_radius


def cluster_words(
    top_words: List[Dict[str, Any]],
    w2v_model: Optional[Word2Vec] = None,
    n_clusters: int = 10,
    random_state: int = 42,
    cluster_seed: Optional[str] = None,
    target_words: Optional[List[str]] = None,
    precomputed_vectors: Optional[np.ndarray] = None,
    return_vectors: bool = False
) -> Any:
    """
    Establishes empirical Medoids first for each cluster using Plan B pairwise dissimilarity,
    then computes synthetic cluster centroids. Supports pre-computed vectors to skip Word2Vec inference.

    Returns:
        (ranked_words_with_clusters, clusters_summary_dict) or
        (ranked_words_with_clusters, clusters_summary_dict, vectors_matrix) if return_vectors is True
    """
    if not top_words:
        if return_vectors:
            return [], {}, np.empty((0, 0), dtype=np.float32)
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
    vector_dim = w2v_model.vector_size if w2v_model else vector_size

    # Fast path: use pre-computed vectors if provided
    word_to_vec: Dict[str, np.ndarray] = {}
    if precomputed_vectors is not None and len(precomputed_vectors) == len(top_words):
        X = np.asarray(precomputed_vectors, dtype=np.float32)
        for i, entry in enumerate(top_words):
            word_to_vec[entry["word"]] = X[i]
    else:
        vectors: List[np.ndarray] = []
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
            word_to_vec[word] = vec
        X = np.array(vectors, dtype=np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    # Perform KMeans clustering to partition vocabulary
    logger.info(f"Running clustering with k={actual_k} on {len(top_words):,} word vectors...")
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

    # Target words set for fast lookup
    targets_set = set(t.lower().strip() for t in (target_words or []))
    if cluster_seed:
        targets_set.add(cluster_seed.lower().strip())

    # Establish Medoids first for each cluster via Plan B pairwise distance matrix
    cluster_anchor_labels: Dict[int, str] = {}
    cluster_medoids: Dict[int, str] = {}
    cluster_medoid_vectors: Dict[int, List[float]] = {}
    cluster_dispersion_radii: Dict[int, float] = {}
    cluster_max_radii: Dict[int, float] = {}
    cluster_totals: Dict[int, int] = {}
    cluster_is_target: Dict[int, bool] = {}
    total_top_words_count = sum(entry.get("count", 0) for entry in top_words)

    for cid, members in clusters_grouped.items():
        # Member vectors
        c_vecs = np.array([word_to_vec[m["word"]] for m in members])
        medoid_idx, disp_radius, max_rad = compute_cluster_medoid_pairwise(c_vecs, metric="cosine")
        
        medoid_entry = members[medoid_idx]
        medoid_word = medoid_entry["word"]
        medoid_vec = c_vecs[medoid_idx].tolist()

        cluster_medoids[cid] = medoid_word
        cluster_medoid_vectors[cid] = medoid_vec
        cluster_dispersion_radii[cid] = round(disp_radius, 4)
        cluster_max_radii[cid] = round(max_rad, 4)

        # Anchor label: highest count member (fallback medoid word)
        members_by_count = sorted(members, key=lambda m: (m.get("count", 0), m.get("combined_score", 0.0)), reverse=True)
        cluster_anchor_labels[cid] = members_by_count[0]["word"]
        cluster_totals[cid] = sum(m.get("count", 0) for m in members)

        # Check if cluster contains target words or cluster seed
        has_target = any(m["word"].lower() in targets_set for m in members)
        cluster_is_target[cid] = has_target

        # Compute distance of every member word to the established medoid via fast BLAS dot product
        if len(c_vecs) > 0:
            norms = np.linalg.norm(c_vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            c_norm = c_vecs / norms
            m_norm = np.array(medoid_vec, dtype=np.float32)
            m_norm_val = np.linalg.norm(m_norm)
            m_norm = m_norm / (m_norm_val if m_norm_val != 0 else 1.0)
            dists_to_medoid = np.clip(1.0 - np.dot(c_norm, m_norm), 0.0, 2.0)
            for m, dist in zip(members, dists_to_medoid):
                m["distance_to_medoid"] = round(float(dist), 4)

    # Assign cluster_label to each top word entry
    for entry in top_words:
        cid = entry["cluster_id"]
        entry["cluster_label"] = cluster_anchor_labels[cid]
        entry["cluster_medoid"] = cluster_medoids[cid]

    # Build outputs/clusters.json structure
    clusters_output: Dict[str, Any] = {}
    
    # Sort cluster IDs: target clusters first (if any), then total count descending
    sorted_cluster_ids = sorted(
        clusters_grouped.keys(),
        key=lambda c: (1 if cluster_is_target[c] else 0, cluster_totals[c]),
        reverse=True
    )

    for cid in sorted_cluster_ids:
        members = clusters_grouped[cid]
        # Sort cluster members: medoid first, then by proximity to medoid (distance_to_medoid ascending)
        members_sorted = sorted(
            members,
            key=lambda m: (
                0 if m["word"] == cluster_medoids[cid] else 1,
                m.get("distance_to_medoid", 1.0),
                -m.get("count", 0)
            )
        )
        member_words = [m["word"] for m in members_sorted]
        c_total = cluster_totals[cid]
        pct_share = round((c_total / total_top_words_count * 100), 2) if total_top_words_count > 0 else 0.0

        centroid_vec = kmeans.cluster_centers_[cid].tolist() if hasattr(kmeans, "cluster_centers_") else []
        clusters_output[str(cid)] = {
            "label": cluster_anchor_labels[cid],
            "medoid": cluster_medoids[cid],
            "medoid_vector": cluster_medoid_vectors[cid],
            "dispersion_radius": cluster_dispersion_radii[cid],
            "max_radius": cluster_max_radii[cid],
            "is_target_cluster": cluster_is_target[cid],
            "words": member_words,
            "total_count": c_total,
            "pct_share": pct_share,
            "centroid_vector": centroid_vec
        }

    if return_vectors:
        return top_words, clusters_output, X
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
    Saves cluster summary dictionary to `outputs/clusters.json` atomically.
    """
    atomic_write_json(output_path, clusters_data)
    logger.info(f"Saved cluster definitions to '{output_path}' ({len(clusters_data)} clusters)")


def build_medoids_summary(clusters_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds Stage 1 medoid summary output structure from cluster dictionary.
    Each cluster's empirical medoid is recorded with dispersion radius and coordinates.
    """
    medoids_output: Dict[str, Any] = {}
    for cid, cdata in clusters_summary.items():
        medoids_output[str(cid)] = {
            "cluster_id": int(cid) if str(cid).isdigit() else cid,
            "cluster_label": cdata.get("label", ""),
            "medoid_word": cdata.get("medoid", cdata.get("label", "")),
            "medoid_vector": cdata.get("medoid_vector", []),
            "dispersion_radius": cdata.get("dispersion_radius", 0.0),
            "max_radius": cdata.get("max_radius", 0.0),
            "member_count": len(cdata.get("words", [])),
            "total_token_count": cdata.get("total_count", 0),
            "pct_share": cdata.get("pct_share", 0.0),
            "top_proximate_words": cdata.get("words", [])[:10],
            "is_target_cluster": cdata.get("is_target_cluster", False),
            "timestamp": datetime.datetime.now().isoformat()
        }
    return medoids_output


def save_medoids_json(medoids_data: Dict[str, Any], output_path: Path) -> None:
    """
    Saves established empirical medoids to `outputs/medoids.json` atomically.
    """
    atomic_write_json(output_path, medoids_data)
    logger.info(f"Saved empirical medoid definitions to '{output_path}' ({len(medoids_data)} medoids)")


def load_target_words(target_file: Path) -> List[str]:
    """Loads optional target words / cluster seeds from target_words.txt."""
    if not target_file.exists():
        return []
    words = []
    with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            clean = line.strip().lower()
            if clean and not clean.startswith("#"):
                words.append(clean)
    return words


def run_clustering_pipeline(
    top_n: int = 1000,
    n_clusters: int = 10,
    raw_csv_path: Optional[Path] = None,
    corpus_path: Optional[Path] = None,
    output_ranked_path: Optional[Path] = None,
    output_clusters_path: Optional[Path] = None,
    output_medoids_path: Optional[Path] = None,
    cluster_seed: Optional[str] = None,
    target_words_path: Optional[Path] = None,
    seed: int = 42,
    use_cache: bool = True,
    vectors_path: Optional[Path] = None,
    vocab_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Orchestrates Stage 1 Medoid Establishment & Clustering pipeline.
    Utilizes decoupled vector caching and BLAS acceleration for sub-second re-clustering.
    """
    raw_csv = raw_csv_path or DEFAULT_RAW_CSV
    corpus_file = corpus_path or DEFAULT_CORPUS_FILE
    ranked_csv = output_ranked_path or DEFAULT_RANKED_CSV
    clusters_json = output_clusters_path or DEFAULT_CLUSTERS_JSON
    medoids_json = output_medoids_path or DEFAULT_MEDOIDS_JSON
    target_file = target_words_path or DEFAULT_TARGET_WORDS_FILE
    vec_path = vectors_path or DEFAULT_VECTORS_NPY
    voc_path = vocab_path or DEFAULT_VOCAB_JSON

    # 1. Read outputs/word_freq_raw.csv
    raw_words = load_raw_frequencies(raw_csv)
    if not raw_words:
        logger.warning("No words found in raw frequency CSV. Creating empty output files.")
        save_ranked_csv([], ranked_csv)
        save_clusters_json({}, clusters_json)
        save_medoids_json({}, medoids_json)
        return {"words_ranked": 0, "clusters_created": 0, "medoids_established": 0}

    # 2. Compute combined scores & rank top N words
    top_words = rank_top_words(raw_words, top_n=top_n)
    top_word_list = [entry["word"] for entry in top_words]

    # Check for valid pre-computed vector cache
    cached_vectors: Optional[np.ndarray] = None
    if use_cache and vec_path.exists() and voc_path.exists():
        try:
            vec_mtime = vec_path.stat().st_mtime
            corpus_mtime = corpus_file.stat().st_mtime if corpus_file.exists() else 0
            raw_mtime = raw_csv.stat().st_mtime if raw_csv.exists() else 0
            if vec_mtime >= corpus_mtime and vec_mtime >= raw_mtime:
                with open(voc_path, "r", encoding="utf-8") as vf:
                    cached_vocab = json.load(vf)
                cached_words = [item["word"] for item in cached_vocab]
                current_words = [entry["word"] for entry in top_words]
                if cached_words == current_words:
                    loaded_X = np.load(vec_path)
                    if len(loaded_X) == len(top_words):
                        cached_vectors = loaded_X
                        logger.info(f"Reusing {len(cached_vectors):,} pre-computed vectors from '{vec_path}' (skipped Word2Vec training).")
        except Exception as ce:
            logger.debug(f"Cache miss reading vectors.npy: {ce}")

    sentences = None
    w2v_model = None
    if cached_vectors is None:
        # 3. Load & tokenize outputs/corpus.txt
        sentences = load_and_tokenize_corpus(corpus_file)

        # 4. Build co-occurrence matrix (sliding window of 5 words)
        build_cooccurrence_matrix(sentences, top_word_list, window_size=5)

        # 5. Train local Word2Vec model (100 dimensions)
        w2v_model = train_word2vec_model(sentences, vector_size=100, window=5, seed=seed)

    # Load target words from file if available (supports both CLI cluster_seed and file-based targets)
    target_words = load_target_words(target_file)
    if cluster_seed:
        logger.info(f"Target cluster seed specified: '{cluster_seed}'")

    # 6. Establish Medoids first via Plan B pairwise distance, then cluster centroids
    ranked_words_with_clusters, clusters_summary, vecs_to_save = cluster_words(
        top_words=top_words,
        w2v_model=w2v_model,
        n_clusters=n_clusters,
        random_state=seed,
        cluster_seed=cluster_seed,
        target_words=target_words,
        precomputed_vectors=cached_vectors,
        return_vectors=True
    )

    # Cache vector representations for instant re-clustering
    if vecs_to_save is not None and len(vecs_to_save) > 0:
        try:
            vec_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(vec_path, vecs_to_save)
            with open(voc_path, "w", encoding="utf-8") as vf:
                json.dump([{"word": tw["word"]} for tw in top_words], vf)
            logger.info(f"Cached {len(vecs_to_save)} word vectors to '{vec_path}'.")
        except Exception as e:
            logger.warning(f"Could not write vector cache: {e}")

    medoids_summary = build_medoids_summary(clusters_summary)

    # 7. Output files
    save_ranked_csv(ranked_words_with_clusters, ranked_csv)
    save_clusters_json(clusters_summary, clusters_json)
    save_medoids_json(medoids_summary, medoids_json)

    # 7b. Execute deterministic co-occurrence graph community detection & bridge concepts
    try:
        collocs_data = None
        col_file = clusters_json.parent / "collocations.json"
        if col_file.exists():
            with open(col_file, "r", encoding="utf-8") as cf:
                collocs_data = json.load(cf)

        graph_sentences = sentences
        if graph_sentences is None and corpus_file.exists():
            graph_sentences = load_and_tokenize_corpus(corpus_file)

        if graph_sentences:
            run_cooccurrence_graph_pipeline(
                sentences=graph_sentences,
                top_words=top_word_list,
                collocations=collocs_data,
                base_dir=clusters_json.parent.parent
            )
            logger.info("Executed co-occurrence graph Louvain community partitioning & bridge analysis.")
    except Exception as ge:
        logger.warning(f"Could not run graph community analysis: {ge}")

    # Print summary table to console highlighting established Medoids
    logger.info("=" * 70)
    logger.info(f"STAGE 1: MEDOIDS ESTABLISHED ({len(medoids_summary)} Clusters | Top {len(ranked_words_with_clusters)} Words)")
    logger.info("=" * 70)
    for cid, data in clusters_summary.items():
        sample_words = ", ".join(data["words"][:6])
        if len(data["words"]) > 6:
            sample_words += f", ... (+{len(data['words']) - 6} more)"
        target_marker = " [TARGET CLUSTER]" if data.get("is_target_cluster") else ""
        medoid_name = data.get("medoid", data["label"])
        radius = data.get("dispersion_radius", 0.0)
        logger.info(f"Cluster {cid} [Medoid: \"{medoid_name}\"{target_marker}]: Radius={radius:.4f} | {data['total_count']:,} tokens ({data['pct_share']}%) | Members: {sample_words}")
    logger.info("=" * 70)

    return {
        "words_ranked": len(ranked_words_with_clusters),
        "clusters_created": len(clusters_summary),
        "medoids_established": len(medoids_summary)
    }


def main():
    parser = argparse.ArgumentParser(
        description="CENTROID Pipeline Stage 1 — Establish empirical Medoids and cluster words using Word2Vec and Plan B Pairwise Dissimilarity."
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
        help="Number of clusters (default: 10)."
    )
    parser.add_argument(
        "--cluster-seed", "--target-cluster",
        dest="cluster_seed",
        type=str,
        default=None,
        help="Specific cluster focus or target word seed to study (e.g. 'deepfakes')."
    )
    parser.add_argument(
        "--target-words",
        type=Path,
        default=DEFAULT_TARGET_WORDS_FILE,
        help=f"Path to target words list (default: {DEFAULT_TARGET_WORDS_FILE})."
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
        "--output-medoids",
        type=Path,
        default=DEFAULT_MEDOIDS_JSON,
        help=f"Path to output medoids JSON (default: {DEFAULT_MEDOIDS_JSON})."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible Word2Vec and KMeans (default: 42)."
    )

    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Optional snapshot identifier to archive results in outputs/snapshots/<snapshot>/."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Bypass vector cache and force Word2Vec retraining."
    )

    args = parser.parse_args()
    try:
        results = run_clustering_pipeline(
            top_n=args.top,
            n_clusters=args.clusters,
            raw_csv_path=args.raw_csv,
            corpus_path=args.corpus,
            output_ranked_path=args.output_ranked,
            output_clusters_path=args.output_clusters,
            output_medoids_path=args.output_medoids,
            cluster_seed=args.cluster_seed,
            target_words_path=args.target_words,
            seed=args.seed,
            use_cache=not args.no_cache
        )
        if args.snapshot:
            import shutil, time
            snapshot_dir = BASE_DIR / "outputs" / "snapshots" / args.snapshot
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            if args.output_ranked.exists():
                shutil.copy2(args.output_ranked, snapshot_dir / "word_freq_ranked.csv")
            if args.output_clusters.exists():
                shutil.copy2(args.output_clusters, snapshot_dir / "clusters.json")
            if args.output_medoids.exists():
                shutil.copy2(args.output_medoids, snapshot_dir / "medoids.json")
            if args.raw_csv.exists():
                shutil.copy2(args.raw_csv, snapshot_dir / "word_freq_raw.csv")
            if args.corpus.exists():
                shutil.copy2(args.corpus, snapshot_dir / "corpus.txt")
            timeline_path = BASE_DIR / "outputs" / "timeline_index.json"
            t_data = []
            if timeline_path.exists():
                try:
                    with open(timeline_path, "r", encoding="utf-8") as tf:
                        t_data = json.load(tf)
                except Exception:
                    t_data = []
            t_data = [item for item in t_data if item.get("id") != args.snapshot]
            t_data.append({
                "id": args.snapshot,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "label": args.snapshot
            })
            with open(timeline_path, "w", encoding="utf-8") as tf:
                json.dump(t_data, tf, indent=2)
            logger.info(f"Saved snapshot '{args.snapshot}' to '{snapshot_dir}'")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Clustering execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
