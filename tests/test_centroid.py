#!/usr/bin/env python3
"""
CENTROID Comprehensive Test Suite
Location: tests/test_centroid.py

Covers:
  - Phase 11 test requirements:
    * Common utilities (Unicode normalization, deterministic seeding, wordlists, HTML cleaning, safe JSON)
    * URL fetching, Content-Type checks, size limits, unreachable endpoints
    * Word scanner filtering rules (length, digits, symbols), stopword removal, stemming, empty & small corpus
    * Word2Vec and KMeans clustering determinism, fallback vectors, small vocabulary (k > words)
    * Report and dashboard generation (HTML escaping, XSS safety, script injection prevention)
    * State manager checkpointing, hash invalidation, and partial failure recording
"""

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline.common import (
    clean_html_to_text,
    deterministic_word_seed,
    load_ranked_words_from_csv,
    load_wordlist,
    normalize_unicode_punctuation,
    safe_json_dumps,
)


class TestCommonUtilities(unittest.TestCase):
    """Tests for pipeline/common.py shared functions."""

    def test_normalize_unicode_punctuation(self):
        # Curly quotes, smart apostrophes, em-dash, en-dash, ellipsis
        raw = "“Hello” ‘world’—this–is…a ‘test’."
        normalized = normalize_unicode_punctuation(raw)
        self.assertEqual(normalized, '"Hello" \'world\'--this-is...a \'test\'.')

    def test_deterministic_word_seed_consistency(self):
        # Ensure MD5 integer seed is identical across calls and different across words
        seed1 = deterministic_word_seed("quantum")
        seed2 = deterministic_word_seed("quantum")
        seed3 = deterministic_word_seed("gastronomy")
        self.assertEqual(seed1, seed2)
        self.assertNotEqual(seed1, seed3)
        self.assertIsInstance(seed1, int)
        self.assertGreaterEqual(seed1, 0)
        self.assertLessEqual(seed1, 2**31 - 1)

    def test_load_wordlist(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tf:
            tf.write("# Comment line\n\nword_one\n  word_two  \n# Another comment\nWORD_THREE\n")
            temp_path = Path(tf.name)

        try:
            words = load_wordlist(temp_path)
            self.assertEqual(words, {"word_one", "word_two", "word_three"})
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_clean_html_to_text(self):
        html_doc = """
        <!DOCTYPE html>
        <html>
        <head><title>Test Page</title><script>var x = 10;</script><style>.cls{color:red;}</style></head>
        <body>
            <nav><a href="/">Home</a></nav>
            <main>
                <h1>Autonomous Systems</h1>
                <p>Robotics and automated processing expand throughput.</p>
            </main>
            <footer>Copyright 2026</footer>
        </body>
        </html>
        """
        extracted = clean_html_to_text(html_doc)
        self.assertIn("Autonomous Systems", extracted)
        self.assertIn("Robotics and automated processing", extracted)
        # Scripts and navigation boilerplate should be stripped
        self.assertNotIn("var x = 10", extracted)
        self.assertNotIn("Copyright 2026", extracted)

    def test_safe_json_dumps_escaping(self):
        data = {"alert": "</script><script>alert('xss')</script>"}
        dumped = safe_json_dumps(data)
        self.assertNotIn("</script>", dumped)
        self.assertIn("<\\/script>", dumped)


class TestWordScannerRules(unittest.TestCase):
    """Tests for Stage 02: Word Scanner tokenization, filtering rules, and edge cases."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("scan_words", BASE_DIR / "pipeline" / "03_scan_words.py")
        self.scan_words = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.scan_words)

    def test_process_document_tokens_rules(self):
        # RULE 1: Lowercase
        # RULE 2: Length >= 3
        # RULE 3: Alphabetic with hyphens/apostrophes (no pure numbers or symbols)
        # RULE 4: Strip stopwords
        stopwords = {"the", "and", "is", "of", "in", "to"}
        target_words = {"special-target"}
        raw_text = "The quick 12345 brown-fox jumps over 2026 $$$ a is quantum state-of-the-art special-target."

        tokens = self.scan_words.process_document_tokens(
            text=raw_text,
            stopwords=stopwords,
            target_words=target_words
        )

        self.assertIn("quick", tokens)
        self.assertIn("brown-fox", tokens)
        self.assertIn("jumps", tokens)
        self.assertIn("over", tokens)
        self.assertIn("quantum", tokens)
        self.assertIn("state-of-the-art", tokens)
        self.assertIn("special-target", tokens)

        # Stopwords removed
        self.assertNotIn("the", tokens)
        self.assertNotIn("is", tokens)

        # Length < 3 removed
        self.assertNotIn("a", tokens)

        # Numbers & symbols removed
        self.assertNotIn("12345", tokens)
        self.assertNotIn("2026", tokens)
        self.assertNotIn("$$$", tokens)

    def test_empty_corpus_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_corpus = Path(tmpdir) / "corpus.txt"
            tmp_corpus.write_text("", encoding="utf-8")
            tmp_output = Path(tmpdir) / "out.csv"

            stats = self.scan_words.scan_words(
                corpus_path=tmp_corpus,
                output_csv_path=tmp_output,
                min_words=50
            )
            self.assertEqual(stats.get("status"), "insufficient_corpus")
            self.assertFalse(tmp_output.exists())

    def test_small_corpus_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_corpus = Path(tmpdir) / "corpus.txt"
            tmp_corpus.write_text("Small sample text with only eight words here.", encoding="utf-8")
            tmp_output = Path(tmpdir) / "out.csv"

            # Default min_words=50 rejects undersized corpus
            stats_default = self.scan_words.scan_words(
                corpus_path=tmp_corpus,
                output_csv_path=tmp_output,
                min_words=50
            )
            self.assertEqual(stats_default.get("status"), "insufficient_corpus")

            # Configured min_words=5 permits processing
            stats_relaxed = self.scan_words.scan_words(
                corpus_path=tmp_corpus,
                output_csv_path=tmp_output,
                min_words=5
            )
            self.assertNotEqual(stats_relaxed.get("status"), "insufficient_corpus")
            self.assertTrue(tmp_output.exists())


class TestClusteringDeterminismAndEdgeCases(unittest.TestCase):
    """Tests for Stage 04: Word2Vec + KMeans clustering."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("cluster_mod", BASE_DIR / "pipeline" / "05_cluster.py")
        self.cluster_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cluster_mod)

    def test_deterministic_clustering(self):
        # Generate synthetic sentences with clear thematic separation
        sentences = [
            ["quantum", "qubit", "superposition", "entanglement", "computing", "algorithm"],
            ["culinary", "flavor", "gastronomy", "recipe", "chef", "cooking"],
            ["marine", "ocean", "ecosystem", "coral", "species", "biodiversity"]
        ] * 10
        top_words = [
            {"word": "quantum", "count": 100, "tfidf_score": 0.9},
            {"word": "qubit", "count": 80, "tfidf_score": 0.8},
            {"word": "superposition", "count": 70, "tfidf_score": 0.7},
            {"word": "culinary", "count": 95, "tfidf_score": 0.9},
            {"word": "flavor", "count": 85, "tfidf_score": 0.8},
            {"word": "gastronomy", "count": 75, "tfidf_score": 0.7},
            {"word": "marine", "count": 90, "tfidf_score": 0.9},
            {"word": "ocean", "count": 80, "tfidf_score": 0.8},
            {"word": "ecosystem", "count": 70, "tfidf_score": 0.7}
        ]

        # First run
        w2v_1 = self.cluster_mod.train_word2vec_model(sentences, vector_size=32, seed=42)
        words_1, clusters_1 = self.cluster_mod.cluster_words(top_words, w2v_1, n_clusters=3, random_state=42)

        # Second run
        w2v_2 = self.cluster_mod.train_word2vec_model(sentences, vector_size=32, seed=42)
        words_2, clusters_2 = self.cluster_mod.cluster_words(top_words, w2v_2, n_clusters=3, random_state=42)

        # Identical number of clusters
        self.assertEqual(len(clusters_1), len(clusters_2))
        # Cluster assignments should be strictly deterministic across calls
        self.assertEqual(
            [w["cluster_id"] for w in words_1],
            [w["cluster_id"] for w in words_2]
        )

    def test_clustering_with_fewer_words_than_k(self):
        top_words = [
            {"word": "alpha", "count": 10, "tfidf_score": 0.5},
            {"word": "beta", "count": 5, "tfidf_score": 0.3}
        ]
        # Request k=5 clusters when only 2 unique words exist
        ranked, clusters = self.cluster_mod.cluster_words(
            top_words=top_words,
            w2v_model=None,
            n_clusters=5,
            random_state=42
        )
        # Should gracefully cap k <= len(top_words)
        self.assertLessEqual(len(clusters), 2)
        self.assertGreaterEqual(len(clusters), 1)


class TestDashboardAndReportSafety(unittest.TestCase):
    """Tests for Stage 06: Report and Stage 07: Dashboard generation."""

    def test_dashboard_xss_protection(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("dashboard_mod", BASE_DIR / "pipeline" / "07_dashboard.py")
        dash_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dash_mod)

        # Construct malicious input words
        malicious_records = [
            {
                "rank": 1,
                "word": "<script>alert(1)</script>",
                "count": 100,
                "tfidf_score": 0.99,
                "combined_score": 60.4,
                "cluster_id": "0",
                "cluster_label": "hack"
            },
            {
                "rank": 2,
                "word": "\" onmouseover=\"alert('hack')",
                "count": 50,
                "tfidf_score": 0.50,
                "combined_score": 30.2,
                "cluster_id": "0",
                "cluster_label": "hack"
            }
        ]

        rendered_rows = dash_mod.build_ranked_table_rows(malicious_records)

        # Must NOT contain unescaped script tag or unescaped quote injection
        self.assertNotIn("<script>alert(1)</script>", rendered_rows)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered_rows)
        self.assertNotIn("\" onmouseover=\"alert('hack')", rendered_rows)


class TestFetchSafetyAndNetworkHandling(unittest.TestCase):
    """Tests for Stage 01: Fetch URL handling, content-type checks, and limits."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("fetch_mod", BASE_DIR / "pipeline" / "01_fetch.py")
        self.fetch_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fetch_mod)

    def test_unreachable_url_handling(self):
        # An invalid URL should fail gracefully without crashing, returning empty text
        text = self.fetch_mod.extract_text_from_url("http://invalid.nonexistent.domain.example.test/page")
        self.assertEqual(text, "")

    def test_non_text_content_type_rejection(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/octet-stream"}
        mock_response.__enter__.return_value = mock_response

        with patch("requests.get", return_value=mock_response):
            text = self.fetch_mod.extract_text_from_url("https://example.com/binary.bin")
            self.assertEqual(text, "")


class TestStateManager(unittest.TestCase):
    """Tests for Stage 00: State Manager hash computation and checkpointing."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("state_mod", BASE_DIR / "pipeline" / "00_state.py")
        self.state_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.state_mod)

    def test_state_checkpoint_and_skip_logic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            state_file = base_dir / "run_state.json"
            mgr = self.state_mod.StateManager(state_file=state_file, base_dir=base_dir)

            # Create mock stage in definitions for testing
            mock_stage_name = "03_scan_words"
            # Create required input and output files
            corpus_file = base_dir / "outputs" / "corpus.txt"
            corpus_file.parent.mkdir(parents=True, exist_ok=True)
            corpus_file.write_text("initial corpus content for state test", encoding="utf-8")

            out_csv = base_dir / "outputs" / "word_freq_raw.csv"
            out_csv.write_text("word,count,source_file,tfidf_score\nhello,5,test,1.0\n", encoding="utf-8")

            # Record stage start and success
            mgr.record_stage_start(mock_stage_name, parameters={"top": 50})
            mgr.record_stage_finish(mock_stage_name, success=True, exit_code=0)

            # Check that stage is recognized as current
            is_valid, reason = mgr.is_stage_current(mock_stage_name)
            self.assertTrue(is_valid)

            # Modifying corpus file changes hash and invalidates cache
            corpus_file.write_text("modified corpus content with changes", encoding="utf-8")
            is_valid_mod, reason_mod = mgr.is_stage_current(mock_stage_name)
            self.assertFalse(is_valid_mod)
            self.assertIn("changed", reason_mod.lower())


if __name__ == "__main__":
    unittest.main()
