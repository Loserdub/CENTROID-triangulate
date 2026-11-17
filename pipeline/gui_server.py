#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CENTROID Pipeline: GUI Server Bridge
Lightweight local HTTP and REST API server providing full interactive control over
the Medoid-First and Super-Centroid workflow from within the browser GUI (dashboard.html).

Endpoints:
  GET  /                     -> Redirects to /dashboard.html
  GET  /api/status           -> Pipeline execution & run_state.json status
  GET  /api/epochs           -> List available historical and active epoch snapshots
  GET  /api/data             -> Real-time bundle of medoids, clusters, drift, & super-centroids
  POST /api/stage/medoids    -> Run Stage 1: Medoid discovery (--cluster-seed, --clusters)
  POST /api/stage/super-centroid -> Run Stage 2: Super Centroid trajectory synthesis
  POST /api/epochs/create    -> Snapshot current corpus into a new epoch folder
"""

import http.server
import importlib.util
import json
import logging
import mimetypes
import os
import re
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from pipeline.common import atomic_write_text, atomic_write_json, validate_snapshot_schema
except ImportError:
    from common import atomic_write_text, atomic_write_json, validate_snapshot_schema

logger = logging.getLogger("CENTROID_GUI_SERVER")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][GUI_SERVER] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent

_PIPELINE_MODULES: Dict[str, Any] = {}


def get_pipeline_module(module_name: str) -> Any:
    """Dynamically loads and caches pipeline script modules for sub-millisecond in-process execution."""
    if module_name in _PIPELINE_MODULES:
        return _PIPELINE_MODULES[module_name]
    script_path = BASE_DIR / "pipeline" / f"{module_name}.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline module not found: {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for module {module_name} at {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _PIPELINE_MODULES[module_name] = mod
    return mod


def get_python_executable() -> str:
    """Finds the local virtual environment Python or falls back to sys.executable."""
    venv_win = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    if venv_win.exists():
        return str(venv_win)
    venv_unix = BASE_DIR / ".venv" / "bin" / "python"
    if venv_unix.exists():
        return str(venv_unix)
    return sys.executable


def run_pipeline_substage(script_name: str, args: list) -> Tuple[bool, str]:
    """Runs a pipeline script in a subprocess and returns (success, combined_output)."""
    py_exec = get_python_executable()
    script_path = BASE_DIR / "pipeline" / script_name
    if not script_path.exists():
        return False, f"Script not found: {script_path}"

    cmd = [py_exec, str(script_path)] + args
    logger.info(f"Executing: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180
        )
        return (proc.returncode == 0), proc.stdout
    except subprocess.TimeoutExpired:
        return False, f"Process timed out after 180s: {script_name}"
    except Exception as e:
        return False, f"Execution failed: {str(e)}"


def get_workbench_bundle(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Constructs a complete real-time bundle including formatted clusters, ranked words,
    target keywords, gap analysis, token statistics, medoids, super-centroids, and drift.
    """
    b_dir = base_dir or BASE_DIR
    try:
        dash_mod = get_pipeline_module("07_dashboard")
    except Exception as e:
        logger.warning(f"Failed to load 07_dashboard module for bundle: {e}")
        dash_mod = None

    # 1. Load raw clusters & convert to formatted cluster_list
    clusters_raw: Dict[str, Any] = {}
    c_file = b_dir / "outputs" / "clusters.json"
    if c_file.exists():
        try:
            with open(c_file, "r", encoding="utf-8") as f:
                clusters_raw = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading clusters.json: {e}")

    cluster_list: List[Dict[str, Any]] = []
    cluster_id_map: Dict[str, str] = {}
    if dash_mod and clusters_raw:
        try:
            cluster_list, cluster_id_map = dash_mod.generate_cluster_dataset(clusters_raw)
        except Exception as e:
            logger.warning(f"Error generating cluster dataset: {e}")

    # 2. Load ranked words
    ranked_words: List[Dict[str, Any]] = []
    r_file = b_dir / "outputs" / "word_freq_ranked.csv"
    if r_file.exists() and dash_mod:
        try:
            ranked_words = dash_mod.load_ranked_words(r_file, top_n=504)
        except Exception as e:
            logger.warning(f"Error loading ranked words: {e}")

    # 3. Load stats
    raw_csv = b_dir / "outputs" / "word_freq_raw.csv"
    corpus_file = b_dir / "outputs" / "corpus.txt"
    total_tokens, unique_words = (0, 0)
    all_counts: Dict[str, int] = {}
    if dash_mod:
        try:
            total_tokens, unique_words, all_counts = dash_mod.load_raw_stats(raw_csv, corpus_file, ranked_words)
        except Exception as e:
            logger.warning(f"Error loading raw stats: {e}")

    core_count = sum(rw.get("count", 0) for rw in ranked_words)
    exempt_count = max(0, total_tokens - core_count)

    # 4. Target words & Gap analysis
    target_words_file = b_dir / "inputs" / "target_words.txt"
    if not target_words_file.exists():
        target_words_file = b_dir / "wordlists" / "target_words.txt"
    target_words: List[str] = []
    if target_words_file.exists() and dash_mod:
        try:
            target_words = dash_mod.load_target_words(target_words_file)
        except Exception as e:
            logger.warning(f"Error loading target words: {e}")

    gaps: List[Dict[str, Any]] = []
    if dash_mod and target_words:
        try:
            gaps = dash_mod.perform_gap_analysis(target_words, all_counts, total_tokens)
        except Exception as e:
            logger.warning(f"Error performing gap analysis: {e}")

    # 5. Sources count
    urls_path = b_dir / "inputs" / "urls.txt"
    sources_count = 0
    if urls_path.exists():
        try:
            with open(urls_path, "r", encoding="utf-8") as uf:
                sources_count = sum(1 for line in uf if line.strip() and not line.strip().startswith("#"))
        except Exception:
            pass

    # 6. Load optional JSON files
    def load_json_opt(rel_path: str) -> Any:
        p = b_dir / rel_path
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    medoids_data = load_json_opt("outputs/medoids.json")
    super_centroids_data = load_json_opt("outputs/super_centroids.json")
    drift_data = load_json_opt("outputs/drift_analysis.json")
    run_state_data = load_json_opt("outputs/run_state.json")
    market_benchmark_data = load_json_opt("outputs/market_benchmark.json")
    trend_forecast_data = load_json_opt("outputs/trend_forecast.json")
    collocations_data = load_json_opt("outputs/collocations.json")
    graph_community_data = load_json_opt("outputs/cooccurrence_graph.json")
    burst_analysis_data = load_json_opt("outputs/burst_analysis.json")

    return {
        "clusters": cluster_list,
        "clusters_raw": clusters_raw,
        "cluster_id_map": cluster_id_map,
        "ranked_words": ranked_words,
        "target_words": target_words,
        "gaps": gaps,
        "medoids": medoids_data,
        "super_centroids": super_centroids_data,
        "drift": drift_data,
        "run_state": run_state_data,
        "market_benchmark": market_benchmark_data,
        "trend_forecast": trend_forecast_data,
        "collocations": collocations_data or [],
        "graph_community": graph_community_data,
        "burst_analysis": burst_analysis_data,
        "stats": {
            "total_tokens": total_tokens,
            "unique_words": unique_words,
            "core_words": core_count,
            "exempt_words": exempt_count,
            "sources_count": sources_count
        }
    }


class CentroidGuiHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving dashboard files and handling REST actions."""

    def __init__(self, *args, **kwargs):
        # Set directory to project base directory
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """Terse HTTP request logging."""
        logger.info(f"{self.address_string()} - {format % args}")

    def send_json(self, status_code: int, data: Dict[str, Any]) -> None:
        """Sends a JSON response with standard CORS headers."""
        response_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """Route GET requests to API or static files."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            # Serve dashboard.html
            self.send_response(302)
            self.send_header("Location", "/dashboard.html")
            self.end_headers()
            return

        if path == "/api/status":
            self.handle_get_status()
            return

        if path == "/api/epochs":
            self.handle_get_epochs()
            return

        if path == "/api/data":
            self.handle_get_data()
            return

        if path == "/api/snapshot/export":
            self.handle_snapshot_export()
            return

        # Fallback to standard static file serving from BASE_DIR
        # Ensure outputs/dashboard.html can also be accessed via /dashboard.html
        if path == "/dashboard.html":
            out_dash = BASE_DIR / "outputs" / "dashboard.html"
            root_dash = BASE_DIR / "dashboard.html"
            target_dash = out_dash if out_dash.exists() else root_dash
            if target_dash.exists():
                with open(target_dash, "rb") as df:
                    content = df.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(content)
                return

        super().do_GET()

    def do_POST(self) -> None:
        """Route POST requests to pipeline actions."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = b""
        if content_length > 0:
            body = self.rfile.read(content_length)

        payload: Dict[str, Any] = {}
        if body:
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception as pe:
                self.send_json(400, {"success": False, "error": f"Invalid JSON payload: {pe}"})
                return

        if path == "/api/stage/medoids":
            self.handle_stage_medoids(payload)
            return

        if path == "/api/stage/super-centroid":
            self.handle_stage_super_centroid(payload)
            return

        if path == "/api/epochs/create":
            self.handle_epoch_create(payload)
            return

        if path == "/api/score-draft":
            self.handle_score_draft(payload)
            return

        if path == "/api/market-benchmark":
            self.handle_market_benchmark(payload)
            return

        if path == "/api/snapshot/upload-multi":
            self.handle_snapshot_upload_multi(payload)
            return

        if path == "/api/trend-forecast":
            self.handle_trend_forecast(payload)
            return

        self.send_json(404, {"success": False, "error": f"Unknown endpoint: {path}"})

    def handle_get_status(self) -> None:
        """Returns the current pipeline run state."""
        state_file = BASE_DIR / "outputs" / "run_state.json"
        state_data: Dict[str, Any] = {"initialized": False}
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as sf:
                    state_data = json.load(sf)
            except Exception as e:
                state_data["read_error"] = str(e)
        self.send_json(200, {"success": True, "status": state_data})

    def handle_get_epochs(self) -> None:
        """Returns list of available epochs in inputs/epochs/ and outputs/timeline_index.json."""
        epochs_dir = BASE_DIR / "inputs" / "epochs"
        epochs_list = []
        if epochs_dir.exists():
            for entry in sorted(epochs_dir.iterdir()):
                if entry.is_dir():
                    raw_files = list(entry.glob("*"))
                    epochs_list.append({
                        "id": entry.name,
                        "path": str(entry),
                        "file_count": len(raw_files)
                    })

        timeline_file = BASE_DIR / "outputs" / "timeline_index.json"
        timeline_data = None
        if timeline_file.exists():
            try:
                with open(timeline_file, "r", encoding="utf-8") as tf:
                    timeline_data = json.load(tf)
            except Exception:
                pass

        self.send_json(200, {
            "success": True,
            "epochs": epochs_list,
            "timeline": timeline_data
        })

    def handle_get_data(self) -> None:
        """Bundles current medoids, super centroids, clusters, and drift data for hot reloading."""
        bundle = get_workbench_bundle(BASE_DIR)
        self.send_json(200, {"success": True, "data": bundle})

    def handle_stage_medoids(self, payload: Dict[str, Any]) -> None:
        """Executes Stage 1 medoid calculation and refreshes dashboard."""
        cluster_seed = payload.get("cluster_seed")
        clusters_k = payload.get("clusters", 10)
        urls_raw = payload.get("urls")
        text_raw = payload.get("text")
        force_ingest = payload.get("force_ingest", False)

        # 1. Parse and persist URLs if provided
        urls: List[str] = []
        if isinstance(urls_raw, list):
            urls = [str(u).strip() for u in urls_raw if str(u).strip() and not str(u).strip().startswith("#")]
        elif isinstance(urls_raw, str):
            urls = [u.strip() for u in urls_raw.splitlines() if u.strip() and not u.strip().startswith("#")]

        re_ingest = bool(urls or text_raw or force_ingest)

        if urls:
            urls_file = BASE_DIR / "inputs" / "urls.txt"
            atomic_write_text(urls_file, "# CENTROID Corpus Sources\n" + "\n".join(urls) + "\n")
            logger.info(f"Persisted {len(urls)} source URL(s) to {urls_file}")

        # 2. Persist direct text into inputs/docs/ if provided
        if text_raw and str(text_raw).strip():
            docs_dir = BASE_DIR / "inputs" / "docs"
            atomic_write_text(docs_dir / "user_corpus.txt", str(text_raw).strip() + "\n")
            logger.info(f"Persisted user text corpus to {docs_dir / 'user_corpus.txt'}")

        # 3. Execute pipeline
        start_time = time.perf_counter()
        if re_ingest:
            # Full end-to-end ingestion, scan, cluster, and dashboard generation
            cmd_args = [
                "--force",
                "--top", "500",
                "--clusters", str(clusters_k),
                "--dashboard"
            ]
            if cluster_seed:
                cmd_args.extend(["--cluster-seed", str(cluster_seed).strip()])

            ok, out = run_pipeline_substage("run.py", cmd_args)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if not ok:
                self.send_json(500, {
                    "success": False,
                    "stage": "medoids",
                    "error": "Failed executing pipeline via run.py",
                    "log": out,
                    "elapsed_ms": round(elapsed_ms, 2)
                })
                return
        else:
            # In-process re-clustering via cached vectors and BLAS medoid kernel
            try:
                cluster_mod = get_pipeline_module("05_cluster")
                dash_mod = get_pipeline_module("07_dashboard")

                clean_seed = str(cluster_seed).strip() if cluster_seed else None
                cluster_res = cluster_mod.run_clustering_pipeline(
                    top_n=500,
                    n_clusters=int(clusters_k),
                    cluster_seed=clean_seed,
                    use_cache=True
                )
                try:
                    drift_mod = get_pipeline_module("08_drift")
                    drift_mod.run_drift_pipeline()
                except Exception as de:
                    logger.warning(f"In-process drift update warning: {de}")
                dash_mod.generate_dashboard(top_n=504)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                out = (
                    f"[INFO] In-memory re-clustering completed in {elapsed_ms:.1f}ms.\n"
                    f"[INFO] Clusters created: {cluster_res.get('clusters_created', clusters_k)}, "
                    f"Medoids established: {cluster_res.get('medoids_established', clusters_k)}.\n"
                    f"[INFO] Dashboard refreshed with synchronized drift radar."
                )
                logger.info(f"In-process re-clustering completed in {elapsed_ms:.1f}ms")
            except Exception as e:
                logger.error(f"In-process re-clustering failed, falling back to subprocess: {e}")
                # Fallback to subprocess
                cmd_args = ["--clusters", str(clusters_k)]
                if cluster_seed:
                    cmd_args.extend(["--cluster-seed", str(cluster_seed).strip()])

                ok, out = run_pipeline_substage("05_cluster.py", cmd_args)
                if not ok:
                    self.send_json(500, {
                        "success": False,
                        "stage": "medoids",
                        "error": "Failed executing 05_cluster.py",
                        "log": out
                    })
                    return

                drift_ok, drift_out = run_pipeline_substage("08_drift.py", [])
                dash_ok, dash_out = run_pipeline_substage("07_dashboard.py", [])
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                out = f"{out}\n{drift_out}\n{dash_out}"

        # 4. Read generated unified workbench bundle
        bundle = get_workbench_bundle(BASE_DIR)

        self.send_json(200, {
            "success": True,
            "stage": "medoids",
            "cluster_seed": cluster_seed,
            "medoids": bundle["medoids"],
            "clusters": bundle["clusters"],
            "clusters_raw": bundle["clusters_raw"],
            "ranked_words": bundle["ranked_words"],
            "target_words": bundle["target_words"],
            "gaps": bundle["gaps"],
            "stats": bundle["stats"],
            "data": bundle,
            "elapsed_ms": round(elapsed_ms, 2),
            "log": out
        })

    def handle_stage_super_centroid(self, payload: Dict[str, Any]) -> None:
        """Executes Stage 2 super-centroid synthesis and refreshes dashboard."""
        start_time = time.perf_counter()
        recency_decay = payload.get("recency_decay", True)
        decay_factor = float(payload.get("decay_factor", 0.85))
        momentum_weight = float(payload.get("momentum_weight", 0.15))

        try:
            drift_mod = get_pipeline_module("08_drift")
            dash_mod = get_pipeline_module("07_dashboard")

            drift_res = drift_mod.run_drift_pipeline(
                use_recency_decay=recency_decay,
                decay_factor=decay_factor,
                momentum_weight=momentum_weight
            )
            dash_mod.generate_dashboard(top_n=504)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            out = (
                f"[INFO] In-memory Stage 2 super-centroid synthesis completed in {elapsed_ms:.1f}ms.\n"
                f"[INFO] Super centroids generated: {len(drift_res.get('super_centroids', {}))}.\n"
                f"[INFO] Dashboard refreshed."
            )
            logger.info(f"In-process super-centroid synthesis completed in {elapsed_ms:.1f}ms")
        except Exception as e:
            logger.error(f"In-process super-centroid failed, falling back to subprocess: {e}")
            cmd_args = [
                "--decay-factor", str(decay_factor),
                "--momentum-weight", str(momentum_weight)
            ]
            if recency_decay:
                cmd_args.append("--recency-decay")
            else:
                cmd_args.append("--no-recency-decay")

            ok, out = run_pipeline_substage("08_drift.py", cmd_args)
            if not ok:
                self.send_json(500, {
                    "success": False,
                    "stage": "super-centroid",
                    "error": "Failed executing 08_drift.py",
                    "log": out
                })
                return

            dash_ok, dash_out = run_pipeline_substage("07_dashboard.py", [])
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            out = f"{out}\n{dash_out}"

        # 3. Read generated unified workbench bundle
        bundle = get_workbench_bundle(BASE_DIR)

        self.send_json(200, {
            "success": True,
            "stage": "super-centroid",
            "super_centroids": bundle["super_centroids"],
            "drift": bundle["drift"],
            "clusters": bundle["clusters"],
            "clusters_raw": bundle["clusters_raw"],
            "ranked_words": bundle["ranked_words"],
            "target_words": bundle["target_words"],
            "stats": bundle["stats"],
            "data": bundle,
            "elapsed_ms": round(elapsed_ms, 2),
            "log": out
        })

    def handle_epoch_create(self, payload: Dict[str, Any]) -> None:
        """Freezes current raw inputs into a new epoch snapshot."""
        epoch_id = payload.get("epoch_id")
        if not epoch_id:
            epoch_id = datetime.utcnow().strftime("epoch_%Y%m%d_%H%M%S")

        epoch_dir = BASE_DIR / "inputs" / "epochs" / str(epoch_id)
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # Copy inputs/raw/ into new epoch
        raw_src = BASE_DIR / "inputs" / "raw"
        copied_files = 0
        if raw_src.exists():
            for item in raw_src.iterdir():
                if item.is_file() and not item.name.startswith("."):
                    shutil.copy2(item, epoch_dir / item.name)
                    copied_files += 1

        self.send_json(200, {
            "success": True,
            "epoch_id": epoch_id,
            "files_copied": copied_files,
            "epoch_path": str(epoch_dir)
        })

    def handle_score_draft(self, payload: Dict[str, Any]) -> None:
        """Scores user-submitted draft text against active clusters and target keywords."""
        text = str(payload.get("text", "")).strip()
        if not text:
            self.send_json(400, {"success": False, "error": "No draft text provided"})
            return

        bundle = get_workbench_bundle(BASE_DIR)
        clusters = bundle.get("clusters", [])
        target_words = [str(w).lower() for w in bundle.get("target_words", [])]

        # Extract normalized words
        tokens = re.findall(r"\b[a-zA-Z0-9_\'-]{2,}\b", text.lower())
        token_set = set(tokens)
        token_count = len(tokens)

        # 1. Cluster Matches
        cluster_matches = []
        covered_clusters = 0
        for c in clusters:
            cid = str(c.get("id", ""))
            clabel = c.get("label", f"Cluster {cid}")
            cwords = set([str(w).lower() for w in c.get("words", [])])
            if c.get("medoid"):
                cwords.add(str(c["medoid"]).lower())

            overlap = token_set.intersection(cwords)
            hit = len(overlap) > 0
            if hit:
                covered_clusters += 1
            cluster_matches.append({
                "id": cid,
                "label": clabel,
                "matched": hit,
                "overlap_count": len(overlap),
                "matched_words": sorted(list(overlap))[:5]
            })

        # 2. Target Words Matches
        matched_targets = []
        missing_targets = []
        for tw in target_words:
            if tw in token_set:
                matched_targets.append(tw)
            else:
                missing_targets.append(tw)

        # 3. Calculate Score (0-100)
        total_clusters = max(1, len(clusters))
        total_targets = max(1, len(target_words)) if target_words else 1

        cluster_score = (covered_clusters / total_clusters) * 55.0
        target_score = (len(matched_targets) / total_targets) * 35.0 if target_words else 35.0
        length_factor = min(1.0, token_count / 150.0) * 10.0
        final_score = int(round(cluster_score + target_score + length_factor))
        final_score = max(0, min(100, final_score))

        # 4. Generate Diagnostic Advice
        if final_score >= 75:
            tier = "OPTIMIZED TOPICAL ALIGNMENT"
            feedback = f"Strong alignment ({final_score}%). Your draft covers {covered_clusters}/{total_clusters} clusters and activates {len(matched_targets)}/{len(target_words)} target keywords."
        elif final_score >= 45:
            tier = "MODERATE TOPICAL FIT"
            missing_labs = [c["label"] for c in cluster_matches if not c["matched"]][:3]
            feedback = f"Moderate fit ({final_score}%). Incorporate concepts from unrepresented clusters [{', '.join(missing_labs)}] and missing target words to deepen relevance."
        else:
            tier = "NEEDS TOPICAL EXPANSION"
            missing_labs = [c["label"] for c in cluster_matches if not c["matched"]][:3]
            feedback = f"Low cluster overlap ({final_score}%). Missing primary anchors like [{', '.join(missing_labs)}]. Weave in target vocabulary to anchor draft to centroid."

        self.send_json(200, {
            "success": True,
            "score": final_score,
            "tier": tier,
            "feedback": feedback,
            "word_count": token_count,
            "covered_clusters": covered_clusters,
            "total_clusters": total_clusters,
            "clusters": cluster_matches,
            "matched_targets": matched_targets,
            "missing_targets": missing_targets
        })

    def handle_market_benchmark(self, payload: Dict[str, Any]) -> None:
        """Executes live SERP market benchmarking and returns differential vector analysis."""
        raw_query = payload.get("query")
        query = str(raw_query).strip() if raw_query else ""
        raw_urls = payload.get("urls")
        urls = []
        if isinstance(raw_urls, list):
            urls = [str(u).strip() for u in raw_urls if str(u).strip()]
        elif isinstance(raw_urls, str) and raw_urls.strip():
            urls = [line.strip() for line in raw_urls.splitlines() if line.strip()]

        max_results = payload.get("max_results", 5)
        try:
            max_results = int(max_results)
        except Exception:
            max_results = 5

        try:
            market_mod = get_pipeline_module("09_market_benchmark")
            result = market_mod.benchmark_market_serp(
                query=query if query else None,
                urls=urls if urls else None,
                max_results=max_results,
                base_dir=BASE_DIR
            )
            bundle = get_workbench_bundle(BASE_DIR)
            self.send_json(200, {
                "success": True,
                "message": f"Market Centroid Benchmark completed for '{result.get('query', query)}'",
                "market_benchmark": result,
                "bundle": bundle
            })
        except Exception as e:
            logger.error(f"Error executing market benchmark: {e}", exc_info=True)
            self.send_json(500, {
                "success": False,
                "error": f"Failed to run Market Centroid Benchmark: {str(e)}"
            })

    def handle_snapshot_export(self) -> None:
        """Exports and returns current snapshot data sheet as JSON with download disposition."""
        try:
            forecast_mod = get_pipeline_module("10_trend_forecast")
            datasheet = forecast_mod.export_snapshot_datasheet(BASE_DIR)
            payload_bytes = json.dumps(datasheet, indent=2).encode("utf-8")
            filename = f"{datasheet.get('snapshot_id', 'centroid_snapshot')}.json"

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(payload_bytes)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(payload_bytes)
        except Exception as e:
            logger.error(f"Error exporting snapshot: {e}", exc_info=True)
            self.send_json(500, {"success": False, "error": f"Export failed: {str(e)}"})

    def handle_snapshot_upload_multi(self, payload: Dict[str, Any]) -> None:
        """Processes multiple uploaded snapshot data sheets to reconstruct a multi-epoch timeline."""
        raw_snapshots = payload.get("snapshots") or []
        if not raw_snapshots:
            self.send_json(400, {
                "success": False,
                "error": "Snapshot upload payload is empty or missing 'snapshots' array."
            })
            return

        snapshots_list: List[Dict[str, Any]] = []
        schema_errors: List[str] = []

        for idx, item in enumerate(raw_snapshots):
            is_valid, err_msg, cleaned = validate_snapshot_schema(item)
            if is_valid and cleaned:
                snapshots_list.append(cleaned)
            else:
                schema_errors.append(f"Snapshot #{idx + 1}: {err_msg}")

        if not snapshots_list:
            self.send_json(400, {
                "success": False,
                "error": "All uploaded snapshots failed schema validation guardrails.",
                "details": schema_errors
            })
            return

        try:
            forecast_mod = get_pipeline_module("10_trend_forecast")
            timeline_res = forecast_mod.reconstruct_multi_snapshot_timeline(snapshots_list)
            self.send_json(200, {
                "success": True,
                "timeline": timeline_res,
                "warnings": schema_errors if schema_errors else []
            })
        except Exception as e:
            logger.error(f"Error processing multi-snapshot upload: {e}", exc_info=True)
            self.send_json(500, {"success": False, "error": f"Multi-snapshot analysis failed: {str(e)}"})

    def handle_trend_forecast(self, payload: Dict[str, Any]) -> None:
        """Generates real-time predictive trend topics and returns updated bundle."""
        horizon = 1.0
        try:
            horizon = float(payload.get("forecast_horizon", 1.0))
        except Exception:
            horizon = 1.0

        try:
            forecast_mod = get_pipeline_module("10_trend_forecast")
            report = forecast_mod.synthesize_predictive_topics(BASE_DIR, forecast_horizon=horizon)
            bundle = get_workbench_bundle(BASE_DIR)
            self.send_json(200, {
                "success": True,
                "trend_forecast": report,
                "bundle": bundle
            })
        except Exception as e:
            logger.error(f"Error synthesizing trend forecast: {e}", exc_info=True)
            self.send_json(500, {"success": False, "error": f"Trend forecast failed: {str(e)}"})


class ThreadedCentroidServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_gui_server(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    """Starts the CENTROID GUI server on the designated host and port."""
    server_address = (host, port)
    try:
        httpd = ThreadedCentroidServer(server_address, CentroidGuiHandler)
    except OSError as e:
        logger.error(f"Could not bind to {host}:{port}: {e}. Retrying on port {port + 1}...")
        port = port + 1
        server_address = (host, port)
        httpd = ThreadedCentroidServer(server_address, CentroidGuiHandler)

    gui_url = f"http://{host}:{port}/dashboard.html"
    logger.info("=" * 68)
    logger.info(f"★ CENTROID NATIVE GUI WORKBENCH ACTIVE")
    logger.info(f"★ Local URL: {gui_url}")
    logger.info(f"★ API Bridge: Stage 1 (Medoids) & Stage 2 (Super-Centroid) Ready")
    logger.info("=" * 68)

    if open_browser:
        def _open():
            time.sleep(0.8)
            logger.info(f"Launching default web browser to: {gui_url}")
            webbrowser.open(gui_url)
        threading.Thread(target=_open, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down CENTROID GUI server...")
        httpd.shutdown()
        httpd.server_close()
        logger.info("CENTROID GUI server stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CENTROID Native GUI Server Bridge")
    parser.add_argument("--host", default="127.0.0.1", help="Host binding (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port binding (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not launch browser automatically")
    args = parser.parse_args()
    start_gui_server(host=args.host, port=args.port, open_browser=not args.no_browser)
