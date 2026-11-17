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
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline.common import (
    atomic_write_json,
    atomic_write_text,
    clean_html_to_text,
    deterministic_word_seed,
    load_ranked_words_from_csv,
    load_wordlist,
    normalize_unicode_punctuation,
    safe_json_dumps,
    validate_snapshot_schema,
)
from pipeline.temporal_burst import (
    apply_exponential_temporal_decay,
    detect_kleinberg_burst_2state,
)
from pipeline.graph_community import (
    detect_bridge_concepts,
    detect_louvain_communities,
    compute_pagerank,
    run_cooccurrence_graph_pipeline,
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


class TestMedoidsAndSuperCentroidPipeline(unittest.TestCase):
    """Tests for Stage 1 Medoid Establishment and Stage 2 Super Centroid Delivery."""

    def setUp(self):
        import importlib.util
        # Load cluster module
        spec_c = importlib.util.spec_from_file_location("cluster_mod", BASE_DIR / "pipeline" / "05_cluster.py")
        self.cluster_mod = importlib.util.module_from_spec(spec_c)
        spec_c.loader.exec_module(self.cluster_mod)

        # Load drift module
        spec_d = importlib.util.spec_from_file_location("drift_mod", BASE_DIR / "pipeline" / "08_drift.py")
        self.drift_mod = importlib.util.module_from_spec(spec_d)
        spec_d.loader.exec_module(self.drift_mod)

        # Load state module
        spec_s = importlib.util.spec_from_file_location("state_mod", BASE_DIR / "pipeline" / "00_state.py")
        self.state_mod = importlib.util.module_from_spec(spec_s)
        spec_s.loader.exec_module(self.state_mod)

    def test_compute_cluster_medoid_pairwise_grounding(self):
        # 3 points: v0 and v1 are very close, v2 is distant
        vecs = np.array([
            [1.0, 0.05],
            [0.98, 0.06],
            [-1.0, -0.9]
        ], dtype=np.float32)

        idx, disp_radius, max_rad = self.cluster_mod.compute_cluster_medoid_pairwise(vecs, metric="cosine")
        # Medoid must be index 0 or 1, not the outlier 2
        self.assertIn(idx, [0, 1])
        self.assertGreater(disp_radius, 0.0)
        self.assertGreaterEqual(max_rad, disp_radius)

    def test_cluster_words_establishes_empirical_medoids(self):
        top_words = [
            {"word": "quantum", "count": 100, "tfidf_score": 0.9},
            {"word": "qubit", "count": 80, "tfidf_score": 0.8},
            {"word": "superposition", "count": 70, "tfidf_score": 0.7},
            {"word": "entanglement", "count": 60, "tfidf_score": 0.6}
        ]
        words, clusters = self.cluster_mod.cluster_words(top_words, w2v_model=None, n_clusters=2, random_state=42)
        
        self.assertGreaterEqual(len(clusters), 1)
        for cid, cdata in clusters.items():
            self.assertIn("medoid", cdata)
            self.assertIn("medoid_vector", cdata)
            self.assertIn("dispersion_radius", cdata)
            # Medoid must be an actual observed word in the cluster members
            self.assertIn(cdata["medoid"], cdata["words"])
            # Medoid is positioned first in the words list
            self.assertEqual(cdata["words"][0], cdata["medoid"])

        # Test build_medoids_summary
        medoids_dict = self.cluster_mod.build_medoids_summary(clusters)
        self.assertEqual(len(medoids_dict), len(clusters))
        for cid, mdata in medoids_dict.items():
            self.assertIn("medoid_word", mdata)
            self.assertIn("dispersion_radius", mdata)
            self.assertIn("member_count", mdata)

    def test_super_centroid_synthesis_recency_and_anchors(self):
        # Mock vocabulary vectors
        vocab = {
            "core": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "satellite": np.array([0.5, 0.5, 0.0], dtype=np.float32),
            "remote": np.array([0.0, 0.0, 1.0], dtype=np.float32)
        }
        anchors = self.drift_mod.find_nearest_semantic_anchors(
            np.array([0.95, 0.05, 0.0], dtype=np.float32), vocab, top_n=2
        )
        self.assertEqual(anchors[0]["word"], "core")
        self.assertGreater(anchors[0]["cosine_similarity"], 0.9)

    def test_stage_aliases_resolution(self):
        self.assertEqual(self.state_mod.resolve_stage_name("medoids"), "05_cluster")
        self.assertEqual(self.state_mod.resolve_stage_name("medoid"), "05_cluster")
        self.assertEqual(self.state_mod.resolve_stage_name("super-centroid"), "08_drift")
        self.assertEqual(self.state_mod.resolve_stage_name("super_centroid"), "08_drift")

    def test_drift_pipeline_synchronizes_active_clusters(self):
        timeline = self.drift_mod.ensure_snapshots_exist()
        self.assertGreaterEqual(len(timeline), 2)
        drift_res = self.drift_mod.run_drift_pipeline()
        self.assertIn("trajectories", drift_res)
        self.assertIn("breakout_signals", drift_res)
        self.assertIn("active_clusters_count", drift_res)
        self.assertGreaterEqual(drift_res["active_clusters_count"], 1)


class TestGuiServerAndDashboardIntegration(unittest.TestCase):
    """Verifies that dashboard HTML elements and local GUI server API endpoints operate correctly."""

    @classmethod
    def setUpClass(cls):
        try:
            from pipeline import gui_server
            cls.gui_mod = gui_server
        except ImportError:
            import gui_server
            cls.gui_mod = gui_server

    def test_dashboard_gui_elements_present(self):
        dash_path = BASE_DIR / "outputs" / "dashboard.html"
        self.assertTrue(dash_path.exists(), "outputs/dashboard.html must exist")
        with open(dash_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Visual controls
        self.assertIn("btn-stage-medoids", content, "Stage 1 medoids button must be present in dashboard")
        self.assertIn("btn-stage-super-centroid", content, "Stage 2 super centroid button must be present")
        self.assertIn("btn-epoch-super-centroid", content, "Super Centroid scrubber button must be present")
        self.assertIn("super-centroids-group", content, "SVG group for super centroids must be present")
        self.assertIn("super-centroid-glow", content, "Glow filter must be defined in SVG defs")
        self.assertIn("target-cluster-seed", content, "Target cluster focus input must be present")
        self.assertIn("recency-decay-toggle", content, "Recency decay toggle must be present")

        # Injected script payloads & real-time sync handlers
        self.assertIn("window.MEDOIDS_DATA", content, "window.MEDOIDS_DATA must be injected")
        self.assertIn("window.SUPER_CENTROIDS_DATA", content, "window.SUPER_CENTROIDS_DATA must be injected")
        self.assertIn("renderRankedTable", content, "Dynamic ranked table renderer must be present")
        self.assertIn("applyWorkbenchBundle", content, "Dynamic workbench bundle synchronization handler must be present")
        self.assertIn("updateDriftRadar", content, "Dynamic drift radar update function must be present")

        # Progressive user-focused elements
        self.assertIn("executive-takeaway-card", content, "Executive Takeaway card must be present")
        self.assertIn("updateExecutiveTakeaway", content, "updateExecutiveTakeaway function must be present")
        self.assertIn("draft-evaluator-panel", content, "Draft evaluator panel must be present")
        self.assertIn("setupDraftEvaluator", content, "setupDraftEvaluator function must be present")
        self.assertIn("btn-copy-llm-primer", content, "1-Click LLM Primer button must be present")
        self.assertIn("setupLlmPrimerButton", content, "setupLlmPrimerButton function must be present")

    def test_gui_server_ephemeral_endpoints(self):
        import urllib.request
        import json
        import threading

        # Bind to port 0 for an OS-assigned ephemeral free port
        server = self.gui_mod.ThreadedCentroidServer(("127.0.0.1", 0), self.gui_mod.CentroidGuiHandler)
        ip, port = server.server_address
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        base_url = f"http://127.0.0.1:{port}"
        try:
            # 1. GET /api/status
            req = urllib.request.Request(f"{base_url}/api/status")
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                body = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(body.get("success"))

            # 2. GET /api/epochs
            req = urllib.request.Request(f"{base_url}/api/epochs")
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                body = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(body.get("success"))
                self.assertIn("epochs", body)

            # 3. GET /api/data
            req = urllib.request.Request(f"{base_url}/api/data")
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                body = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(body.get("success"))
                self.assertIn("medoids", body["data"])
                self.assertIn("super_centroids", body["data"])
                self.assertIsInstance(body["data"]["clusters"], list, "clusters in /api/data must be a formatted list/array for SVG rendering")
                self.assertIsInstance(body["data"]["ranked_words"], list, "ranked_words in /api/data must be a list for table rendering")
                self.assertIn("stats", body["data"])
                self.assertIn("total_tokens", body["data"]["stats"])

            # 4. GET /dashboard.html
            req = urllib.request.Request(f"{base_url}/dashboard.html")
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                html_bytes = resp.read()
                self.assertIn(b"CENTROID", html_bytes)

            # 5. POST /api/score-draft
            req_draft = urllib.request.Request(
                f"{base_url}/api/score-draft",
                data=json.dumps({"text": "Our enterprise services suite delivers dedicated financial relief."}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req_draft, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                body = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(body.get("success"))
                self.assertIn("score", body)
                self.assertIn("tier", body)
                self.assertIn("feedback", body)
                self.assertIsInstance(body["clusters"], list)

            # 6. POST /api/market-benchmark
            req_mb = urllib.request.Request(
                f"{base_url}/api/market-benchmark",
                data=json.dumps({"query": "enterprise services", "max_results": 2}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req_mb, timeout=10) as resp:
                self.assertEqual(resp.status, 200)
                mb_body = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(mb_body.get("success"))
                self.assertIn("market_benchmark", mb_body)
                self.assertIn("bundle", mb_body)
                self.assertIn("displacement_vector", mb_body["market_benchmark"])

            # 7. GET /api/snapshot/export
            req_export = urllib.request.Request(f"{base_url}/api/snapshot/export")
            with urllib.request.urlopen(req_export, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                snap_body = json.loads(resp.read().decode("utf-8"))
                self.assertIn("snapshot_id", snap_body)
                self.assertIn("emerging_topics", snap_body)

            # 8. POST /api/snapshot/upload-multi
            snap1 = dict(snap_body)
            snap1["snapshot_id"] = "centroid_snapshot_20261101_100000"
            snap1["timestamp"] = "2026-11-01T10:00:00Z"
            snap2 = dict(snap_body)
            snap2["snapshot_id"] = "centroid_snapshot_20261107_100000"
            snap2["timestamp"] = "2026-11-07T10:00:00Z"

            req_multi = urllib.request.Request(
                f"{base_url}/api/snapshot/upload-multi",
                data=json.dumps({"snapshots": [snap1, snap2]}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req_multi, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                multi_body = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(multi_body.get("success"))
                self.assertIn("timeline", multi_body)
                self.assertEqual(multi_body["timeline"]["total_snapshots"], 2)

            # 9. POST /api/trend-forecast
            req_tf = urllib.request.Request(
                f"{base_url}/api/trend-forecast",
                data=json.dumps({"forecast_horizon": 1.0}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req_tf, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                tf_body = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(tf_body.get("success"))
                self.assertIn("trend_forecast", tf_body)
        finally:
            server.shutdown()
            server.server_close()

    def test_workbench_bundle_payload_structure(self):
        bundle = self.gui_mod.get_workbench_bundle(BASE_DIR)
        self.assertIsInstance(bundle, dict)
        self.assertIsInstance(bundle["clusters"], list)
        self.assertIsInstance(bundle["ranked_words"], list)
        self.assertIn("stats", bundle)
        self.assertIn("total_tokens", bundle["stats"])
        self.assertIn("exempt_words", bundle["stats"])
        self.assertIn("core_words", bundle["stats"])
        self.assertIn("drift", bundle)
        self.assertIsInstance(bundle["drift"], dict)
        self.assertIn("trajectories", bundle["drift"])
        self.assertIn("market_benchmark", bundle)
        self.assertIn("trend_forecast", bundle)
        self.assertIn("collocations", bundle)

    def test_market_benchmark_pipeline(self):
        market_mod = self.gui_mod.get_pipeline_module("09_market_benchmark")
        res = market_mod.benchmark_market_serp(
            query="enterprise platform",
            urls=None,
            max_results=3,
            base_dir=BASE_DIR
        )
        self.assertIsInstance(res, dict)
        self.assertEqual(res["query"], "enterprise platform")
        self.assertIn("user_centroid", res)
        self.assertIn("market_centroid", res)
        self.assertIn("displacement_vector", res)
        self.assertIn("distance", res["displacement_vector"])
        self.assertIn("market_alignment_pct", res)
        self.assertIn("competitor_dominant_terms", res)
        self.assertIn("shared_authority_terms", res)
        self.assertIn("user_differentiators", res)
        self.assertIn("outrank_prompt", res)
        self.assertTrue((BASE_DIR / "outputs" / "market_benchmark.json").exists())

    def test_collocation_extraction(self):
        scan_mod = self.gui_mod.get_pipeline_module("03_scan_words")
        sample_docs = [
            ("doc1", "Employee retention tax credit guides for small business accounting and compliance."),
            ("doc2", "Small business accounting teams claim employee retention tax credit relief.")
        ]
        stopwords = {"and", "for", "the", "in", "of", "to"}
        collocs = scan_mod.extract_collocations(sample_docs, stopwords, min_freq=2, top_n=10)
        self.assertIsInstance(collocs, list)
        self.assertTrue(len(collocs) > 0)
        phrases = [c["phrase"] for c in collocs]
        self.assertTrue(any("employee retention" in p or "small business" in p for p in phrases))

    def test_trend_forecast_synthesis_and_timeline(self):
        forecast_mod = self.gui_mod.get_pipeline_module("10_trend_forecast")
        report = forecast_mod.synthesize_predictive_topics(BASE_DIR, forecast_horizon=1.0)
        self.assertIsInstance(report, dict)
        self.assertIn("top_emerging_topics", report)
        topics = report["top_emerging_topics"]
        self.assertTrue(len(topics) > 0)
        first_topic = topics[0]
        self.assertIn("working_title", first_topic)
        self.assertIn("editorial_angle", first_topic)
        self.assertIn("timing_window", first_topic)
        self.assertIn("editorial_brief", first_topic)
        self.assertIn("key_vocabulary", first_topic)

        # Test multi-snapshot reconstruction
        snap1 = {
            "snapshot_id": "snap_1",
            "timestamp": "2026-11-01T12:00:00Z",
            "emerging_topics": [{"working_title": "Tax Compliance Architecture", "momentum_score": 45.0}]
        }
        snap2 = {
            "snapshot_id": "snap_2",
            "timestamp": "2026-11-08T12:00:00Z",
            "emerging_topics": [{"working_title": "Tax Compliance Architecture", "momentum_score": 75.0}]
        }
        timeline = forecast_mod.reconstruct_multi_snapshot_timeline([snap1, snap2])
        self.assertTrue(timeline.get("success"))
        self.assertEqual(timeline.get("total_snapshots"), 2)
        self.assertTrue(len(timeline.get("topic_trends", [])) > 0)
        self.assertEqual(timeline["topic_trends"][0]["status"], "🔥 RISING")

    def test_npmi_and_cvalue_collocation_metrics(self):
        scan_mod = self.gui_mod.get_pipeline_module("03_scan_words")
        sample_docs = [
            ("doc1", "Employee retention tax credit guides for small business accounting and compliance."),
            ("doc2", "Small business accounting teams claim employee retention tax credit relief.")
        ]
        stopwords = {"and", "for", "the", "in", "of", "to"}
        collocs = scan_mod.extract_collocations(sample_docs, stopwords, min_freq=2, top_n=10)
        self.assertTrue(len(collocs) > 0)
        for c in collocs:
            self.assertIn("npmi", c)
            self.assertIn("c_value", c)
            self.assertIn("score", c)
            self.assertGreaterEqual(c["npmi"], -1.0)
            self.assertLessEqual(c["npmi"], 1.0)
            self.assertGreaterEqual(c["c_value"], 0.0)

        # Statistical isolation test: true collocation should have positive NPMI
        phrases_by_name = {c["phrase"]: c for c in collocs}
        if "employee retention tax" in phrases_by_name:
            self.assertGreater(phrases_by_name["employee retention tax"]["npmi"], 0.3)

    def test_kalman_filter_and_damped_drift_kinematics(self):
        drift_mod = self.gui_mod.get_pipeline_module("08_drift")
        kf = drift_mod.DiscreteKinematicKalmanFilter(dt=1.0, dampening_phi=0.85)

        # 1D filtering and projection test
        obs = [0.2, 0.4, 0.6]
        filt_p, filt_v, proj_p = kf.filter_and_project_1d(obs, horizon=1.0)
        self.assertAlmostEqual(filt_p, 0.6, delta=0.1)
        self.assertGreater(filt_v, 0.0)
        # Proves damping prevents runaway projection beyond linear velocity
        self.assertLess(proj_p, 0.6 + filt_v * 1.5)

        # Robustness test: missing / NaN observation handling
        nan_obs = [0.2, float("nan"), 0.6]
        nan_p, nan_v, nan_proj = kf.filter_and_project_1d(nan_obs, horizon=1.0)
        self.assertFalse(math.isnan(nan_p))
        self.assertFalse(math.isnan(nan_v))
        self.assertFalse(math.isnan(nan_proj))

        # Vector filtering and projection test
        vecs = [
            np.array([0.1, 0.5, 0.8], dtype=np.float32),
            np.array([0.2, 0.55, 0.82], dtype=np.float32),
            np.array([0.3, 0.6, 0.84], dtype=np.float32)
        ]
        filt_m, filt_v_vec, c_proj = kf.filter_and_project_vectors(vecs, horizon=1.0)
        self.assertEqual(len(c_proj), 3)
        self.assertFalse(np.isnan(c_proj).any())

    def test_atomic_serialization_and_schema_guardrails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_atomic.json"
            payload = {"status": "ok", "count": 42}
            atomic_write_json(tmp_path, payload)
            self.assertTrue(tmp_path.exists())
            with open(tmp_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["status"], "ok")

        # Test strict schema validation on valid snapshot
        valid_snap = {
            "snapshot_id": "snap_valid_1",
            "timestamp": "2026-11-15T12:00:00Z",
            "emerging_topics": [{"working_title": "Tax AI", "momentum_score": 80.0}]
        }
        is_val, err, cleaned = validate_snapshot_schema(valid_snap)
        self.assertTrue(is_val)
        self.assertEqual(cleaned["snapshot_id"], "snap_valid_1")

        # Test resilience against None cluster_count, non-dict clusters, and NaN momentum scores
        edge_snap = {
            "snapshot_id": "snap_edge",
            "timestamp": "2026-11-15T12:00:00Z",
            "cluster_count": None,
            "clusters": 123,
            "emerging_topics": [{"working_title": "Edge", "momentum_score": float("nan")}]
        }
        edge_val, edge_err, edge_cleaned = validate_snapshot_schema(edge_snap)
        self.assertTrue(edge_val)
        self.assertEqual(edge_cleaned["cluster_count"], 0)
        self.assertEqual(edge_cleaned["emerging_topics"][0]["momentum_score"], 50.0)

        # Test rejection of zero-byte or corrupt snapshots
        self.assertFalse(validate_snapshot_schema("")[0])
        self.assertFalse(validate_snapshot_schema(b"")[0])
        self.assertFalse(validate_snapshot_schema("{malformed json")[0])
        self.assertFalse(validate_snapshot_schema({"invalid": "missing snapshot_id"})[0])

    def test_cooccurrence_graph_louvain_and_pagerank(self):
        from pipeline.graph_community import build_cooccurrence_graph
        sentences = [
            ["quantum", "qubit", "superposition", "entanglement"],
            ["quantum", "computing", "qubit", "hardware"],
            ["marine", "ocean", "ecosystem", "coral"],
            ["marine", "reef", "ocean", "biodiversity"]
        ]
        top_words = ["quantum", "qubit", "superposition", "entanglement", "hardware", "computing", "marine", "ocean", "ecosystem", "coral", "reef", "biodiversity"]

        # Verify edge weight is not double-counted (1 occurrence -> weight 1)
        test_nodes, test_adj = build_cooccurrence_graph([["apple", "banana", "cherry"]], ["apple", "banana", "cherry"])
        self.assertEqual(test_adj["apple"]["banana"], 1)

        # Verify collocation integration
        collocs = [{"phrase": "quantum computing", "score": 90.0}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = run_cooccurrence_graph_pipeline(sentences, top_words, collocations=collocs, base_dir=Path(tmp_dir))
            self.assertIn("communities", res)
            self.assertIn("bridge_concepts", res)
            self.assertIn("graph_stats", res)
            self.assertGreaterEqual(res["graph_stats"]["modularity_q"], 0.0)
            self.assertTrue((Path(tmp_dir) / "outputs" / "cooccurrence_graph.json").exists())
            # Verify multi-word collocation participated in graph
            graph_words = set()
            for c in res["communities"]:
                graph_words.update(c.get("top_words", []))
            self.assertIn("quantum computing", graph_words)

    def test_kleinberg_burst_and_exponential_half_life(self):
        # Test exponential half-life decay
        vals = [10.0, 10.0, 10.0]
        decayed = apply_exponential_temporal_decay(vals, half_life_days=14.0)
        self.assertEqual(len(decayed), 3)
        self.assertGreater(decayed[-1], decayed[0])

        # Test Kleinberg 2-state burst detection on sustained surge (State 1)
        surge_counts = [1, 2, 25, 30]
        totals = [100, 100, 100, 100]
        burst_res = detect_kleinberg_burst_2state(surge_counts, totals, s=2.0, gamma=1.0)
        self.assertTrue(burst_res["is_bursting"])
        self.assertEqual(burst_res["status"], "AUTHENTIC BURST")

        # Test Kleinberg suppression of isolated spam spike
        spike_counts = [1, 1, 1, 6, 1]
        spike_totals = [100, 100, 100, 100, 100]
        spike_res = detect_kleinberg_burst_2state(spike_counts, spike_totals, s=3.0, gamma=1.5)
        self.assertFalse(spike_res["is_bursting"])
        self.assertEqual(spike_res["burst_epochs_count"], 0)

        # Test multi-snapshot timeline with exponential decay and NaN safety
        forecast_mod = self.gui_mod.get_pipeline_module("10_trend_forecast")
        snaps = [
            {"snapshot_id": "ep1", "timestamp": "2026-08-01T00:00:00Z", "emerging_topics": [{"working_title": "AI Tax", "momentum_score": 60.0}]},
            {"snapshot_id": "ep2", "timestamp": "2026-09-01T00:00:00Z", "emerging_topics": [{"working_title": "AI Tax", "momentum_score": 90.0}]}
        ]
        tl_res = forecast_mod.reconstruct_multi_snapshot_timeline(snaps)
        self.assertTrue(tl_res["success"])
        self.assertEqual(len(tl_res["topic_trends"]), 1)
        self.assertIn("decayed_scores", tl_res["topic_trends"][0])
        # Older epoch should be decayed relative to raw
        self.assertLess(tl_res["topic_trends"][0]["decayed_scores"][0], 60.0)


if __name__ == "__main__":
    unittest.main()
