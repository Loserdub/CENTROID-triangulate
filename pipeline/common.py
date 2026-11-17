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
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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


def atomic_write_json(target_path: Union[Path, str], data: Any, indent: int = 2) -> None:
    """
    Atomically serializes data to a target JSON file using temp-and-replace semantics
    (NamedTemporaryFile on the same parent directory + os.replace) to prevent zero-byte corruptions.
    """
    path = Path(target_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".tmp_{path.stem}_",
        suffix=".tmp",
        delete=False,
        encoding="utf-8"
    )
    temp_path = Path(temp_file.name)
    try:
        json.dump(data, temp_file, indent=indent, ensure_ascii=False)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()  # Required on Windows before os.replace
        os.replace(temp_path, path)
    except Exception:
        temp_file.close()
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def atomic_write_text(target_path: Union[Path, str], text: str) -> None:
    """
    Atomically writes text to a target file using temp-and-replace semantics.
    """
    path = Path(target_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".tmp_{path.stem}_",
        suffix=".tmp",
        delete=False,
        encoding="utf-8"
    )
    temp_path = Path(temp_file.name)
    try:
        temp_file.write(text)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()
        os.replace(temp_path, path)
    except Exception:
        temp_file.close()
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def validate_snapshot_schema(data: Any) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Strict schema validation for snapshot ingestion to prevent zero-byte corruptions
    and unhandled crashes.

    Returns:
        (is_valid: bool, error_msg: str, cleaned_snapshot: Optional[Dict])
    """
    if data is None:
        return False, "Payload is null or empty.", None

    if isinstance(data, (bytes, bytearray)):
        try:
            data = data.decode("utf-8")
        except Exception as e:
            return False, f"Byte payload cannot be decoded as UTF-8: {e}", None

    if isinstance(data, str):
        stripped = data.strip()
        if not stripped:
            return False, "Snapshot data is a zero-byte or empty string.", None
        try:
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                return False, f"Snapshot JSON root must be an object/dict, got {type(parsed).__name__}.", None
            data = parsed
        except json.JSONDecodeError as jde:
            return False, f"Invalid JSON syntax in snapshot payload: {jde.msg} (line {jde.lineno}, col {jde.colno})", None

    if not isinstance(data, dict):
        return False, f"Expected dictionary for snapshot data, received {type(data).__name__}.", None

    snapshot_id = data.get("snapshot_id")
    if not snapshot_id or not isinstance(snapshot_id, str) or not snapshot_id.strip():
        alt_id = data.get("id")
        if alt_id and isinstance(alt_id, str) and alt_id.strip():
            snapshot_id = alt_id.strip()
        else:
            return False, "Missing or invalid required string property 'snapshot_id'.", None

    timestamp = data.get("timestamp")
    if not timestamp or not isinstance(timestamp, str) or not timestamp.strip():
        return False, "Missing or invalid required string property 'timestamp'.", None

    cleaned_topics = []
    if "emerging_topics" in data:
        topics = data["emerging_topics"]
        if not isinstance(topics, list):
            return False, f"'emerging_topics' must be a list, received {type(topics).__name__}.", None
        for i, t in enumerate(topics):
            if not isinstance(t, dict):
                return False, f"Topic at index {i} in 'emerging_topics' must be a dictionary.", None
            title = t.get("working_title") or t.get("cluster_label") or t.get("title")
            if not title:
                continue
            try:
                score = float(t.get("momentum_score", 50.0))
                if math.isnan(score) or math.isinf(score):
                    score = 50.0
            except (ValueError, TypeError):
                score = 50.0

            cleaned_topic = dict(t)
            cleaned_topic["working_title"] = str(title)
            cleaned_topic["momentum_score"] = score
            cleaned_topics.append(cleaned_topic)

    # Safe extraction of cluster_count
    raw_cluster_count = data.get("cluster_count")
    if raw_cluster_count is not None:
        try:
            val = int(float(raw_cluster_count))
            cluster_count = max(0, val) if not (math.isnan(val) or math.isinf(val)) else 0
        except (ValueError, TypeError):
            cluster_count = 0
    else:
        clusters_obj = data.get("clusters")
        if isinstance(clusters_obj, (dict, list)):
            cluster_count = len(clusters_obj)
        else:
            cluster_count = 0

    # Safe extraction of total_vocabulary
    raw_vocab = data.get("total_vocabulary")
    if raw_vocab is not None:
        try:
            val = int(float(raw_vocab))
            total_vocab = max(0, val) if not (math.isnan(val) or math.isinf(val)) else 0
        except (ValueError, TypeError):
            total_vocab = 0
    else:
        ranked_obj = data.get("ranked_sample")
        if isinstance(ranked_obj, list):
            total_vocab = len(ranked_obj)
        else:
            total_vocab = 0

    cleaned: Dict[str, Any] = {
        "snapshot_id": str(snapshot_id).strip(),
        "timestamp": str(timestamp).strip(),
        "corpus_name": str(data.get("corpus_name", "Active Domain Monitor")),
        "cluster_count": cluster_count,
        "total_vocabulary": total_vocab,
        "collocations": data.get("collocations", []) if isinstance(data.get("collocations"), list) else [],
        "emerging_topics": cleaned_topics,
        "sunset_topics": data.get("sunset_topics", []) if isinstance(data.get("sunset_topics"), list) else [],
        "ranked_sample": data.get("ranked_sample", []) if isinstance(data.get("ranked_sample"), list) else []
    }
    for k, v in data.items():
        if k not in cleaned:
            cleaned[k] = v

    return True, "", cleaned


def compute_cluster_2d_coordinates(
    clusters: Union[Dict[str, Any], List[Dict[str, Any]]],
    min_dist: float = 0.14
) -> Dict[str, Tuple[float, float]]:
    """
    Computes distinct, non-overlapping 2D (velocity, density) coordinates for clusters.
    Uses PCA projection on high-dimensional centroid/medoid embeddings when available,
    paired with a deterministic spring-repulsion pass (min_dist >= 0.14) to guarantee
    that no two cluster centers sit on top of each other.

    Coordinates are mapped to the manifold:
      Velocity (X): [0.20, 0.88]
      Density (Y):  [0.48, 0.86]
    """
    items = []
    if isinstance(clusters, dict):
        items = list(clusters.items())
    elif isinstance(clusters, list):
        items = [(c.get("raw_id", c.get("id", str(i))), c) for i, c in enumerate(clusters)]

    k = len(items)
    if k == 0:
        return {}
    if k == 1:
        return {str(items[0][0]): (0.54, 0.68)}

    vectors = []
    has_vectors = True
    try:
        import numpy as np
        from sklearn.decomposition import PCA

        for _, cdata in items:
            vec = cdata.get("centroid_vector") or cdata.get("medoid_vector")
            if vec and isinstance(vec, (list, tuple, np.ndarray)) and len(vec) >= 2:
                vectors.append(list(vec))
            else:
                has_vectors = False
                break

        if has_vectors and len(vectors) == k:
            pca = PCA(n_components=2, random_state=42)
            raw_2d = pca.fit_transform(vectors)
            x_min, x_max = float(raw_2d[:, 0].min()), float(raw_2d[:, 0].max())
            y_min, y_max = float(raw_2d[:, 1].min()), float(raw_2d[:, 1].max())

            x_span = max(1e-6, x_max - x_min)
            y_span = max(1e-6, y_max - y_min)

            x_norm = 0.22 + ((raw_2d[:, 0] - x_min) / x_span) * 0.64
            y_norm = 0.50 + ((raw_2d[:, 1] - y_min) / y_span) * 0.34
            coords = np.column_stack([x_norm, y_norm])
        else:
            raise ValueError("Incomplete vectors, use analytical dispersion")
    except Exception:
        import numpy as np
        coords = np.zeros((k, 2), dtype=float)
        golden_angle = 2.399963229728653  # golden angle in radians
        for i in range(k):
            r = 0.12 + 0.20 * math.sqrt((i + 1) / max(1, k))
            theta = i * golden_angle
            coords[i, 0] = 0.54 + r * math.cos(theta) * 1.4
            coords[i, 1] = 0.67 + r * math.sin(theta) * 0.75

    # Spring repulsion pass to strictly prevent visual overlap
    try:
        import numpy as np
        for _ in range(80):
            for i in range(k):
                for j in range(i + 1, k):
                    delta = coords[i] - coords[j]
                    dist = float(np.linalg.norm(delta))
                    if dist < min_dist:
                        push = (delta / max(1e-6, dist)) * (min_dist - dist) * 0.5
                        coords[i] += push
                        coords[j] -= push
            coords[:, 0] = np.clip(coords[:, 0], 0.18, 0.88)
            coords[:, 1] = np.clip(coords[:, 1], 0.48, 0.86)
    except Exception:
        pass

    result: Dict[str, Tuple[float, float]] = {}
    for i, (cid, _) in enumerate(items):
        vx = round(float(coords[i, 0]), 2)
        vy = round(float(coords[i, 1]), 3)
        result[str(cid)] = (vx, vy)

    return result
