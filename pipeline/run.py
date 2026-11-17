#!/usr/bin/env python3
"""
CENTROID — Single Entry Point Pipeline Runner
Specification: AGENT.md Section 7 (ENTRY POINT) & Section 4

Description:
  Wires together CENTROID pipeline stages 01 through 07 with state tracking:
    Stage 01a: Ingest URLs (01_fetch.py) -> outputs/corpus.txt
    Stage 01b: Ingest Docs/PDFs (02_extract_pdf.py) -> outputs/corpus.txt
    Stage 02:  Word Scanner (03_scan_words.py) -> outputs/word_freq_raw.csv
    Stage 03:  Web Expand (optional, 04_web_expand.py) + Re-scan -> outputs/word_freq_raw.csv
    Stage 04:  Rank + Cluster (05_cluster.py) -> outputs/word_freq_ranked.csv, outputs/clusters.json
    Stage 05/06: Report Generator (06_report.py) -> outputs/REPORT.md
    Stage 07:  Interactive Dashboard (optional via --dashboard, 07_dashboard.py) -> outputs/dashboard.html

Features:
  - Incremental Checkpointing (pipeline/00_state.py):
      Skips completed stages whose input files haven't changed (printing '[skip] stage already current').
  - Partial Resumption:
      Records failure points so re-runs resume from the failed stage.
  - Isolated Stage Runner:
      `--stage NAME` executes exactly one named stage using existing outputs/.
  - Telemetry:
      Prints a formatted pipeline status table at the start of every invocation.

Usage:
  # Full pipeline
  python run.py

  # Force all stages (ignore cache)
  python run.py --force

  # Run single stage in isolation
  python run.py --stage scan
  python run.py --stage cluster --clusters 8 --top 500

  # With options
  python run.py --top 500 --clusters 8 --stem --web-expand --dashboard

  # Output only (skip re-fetch / re-scan)
  python run.py --report-only --dashboard
"""

import argparse
import importlib.util
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Base directory is the project root (CENTROID/)
BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_DIR = BASE_DIR / "pipeline"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Pipeline stage script paths
SCRIPT_00_STATE = PIPELINE_DIR / "00_state.py"
SCRIPT_01_FETCH = PIPELINE_DIR / "01_fetch.py"
SCRIPT_02_EXTRACT = PIPELINE_DIR / "02_extract_pdf.py"
SCRIPT_03_SCAN = PIPELINE_DIR / "03_scan_words.py"
SCRIPT_04_EXPAND = PIPELINE_DIR / "04_web_expand.py"
SCRIPT_05_CLUSTER = PIPELINE_DIR / "05_cluster.py"
SCRIPT_06_REPORT = PIPELINE_DIR / "06_report.py"
SCRIPT_07_DASHBOARD = PIPELINE_DIR / "07_dashboard.py"
SCRIPT_08_DRIFT = PIPELINE_DIR / "08_drift.py"

# Key output paths
PATH_CORPUS = BASE_DIR / "outputs" / "corpus.txt"
PATH_RAW_CSV = BASE_DIR / "outputs" / "word_freq_raw.csv"
PATH_RANKED_CSV = BASE_DIR / "outputs" / "word_freq_ranked.csv"
PATH_CLUSTERS_JSON = BASE_DIR / "outputs" / "clusters.json"
PATH_REPORT_MD = BASE_DIR / "outputs" / "REPORT.md"
PATH_DASHBOARD_HTML = BASE_DIR / "outputs" / "dashboard.html"
PATH_DRIFT_JSON = BASE_DIR / "outputs" / "drift_analysis.json"
PATH_RUN_STATE = BASE_DIR / "outputs" / "run_state.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("CENTROID_RUNNER")


def load_state_manager_module():
    """
    Dynamically loads pipeline/00_state.py to access StateManager and helpers.
    """
    if not SCRIPT_00_STATE.exists():
        return None, None, None, None

    try:
        spec = importlib.util.spec_from_file_location("pipeline_00_state", str(SCRIPT_00_STATE))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return (
                getattr(mod, "StateManager", None),
                getattr(mod, "resolve_stage_name", None),
                getattr(mod, "get_all_stage_names", None),
                getattr(mod, "STAGE_DEFINITIONS", {})
            )
    except Exception as e:
        logger.warning(f"Could not load state module from {SCRIPT_00_STATE}: {e}")

    return None, None, None, None


StateManagerClass, resolve_stage_name_func, get_all_stage_names_func, STAGE_DEFINITIONS = load_state_manager_module()


# Global tracker for pipeline run telemetry
GLOBAL_TRACKER = {
    "has_warnings": False,
    "start_time": None,
}


def run_stage_command(script_path: Path, args: List[str], stage_display_name: str) -> Tuple[bool, int]:
    """
    Executes a pipeline stage script as a subprocess using the current Python interpreter.
    Streams output in real-time while monitoring for warnings and errors.
    Returns (success_bool, exit_code).
    """
    if not script_path.exists():
        logger.warning(f"Stage script '{script_path.name}' not found at {script_path}. Skipping.")
        GLOBAL_TRACKER["has_warnings"] = True
        return False, 127

    venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = BASE_DIR / ".venv" / "bin" / "python"
    py_exec = str(venv_python) if venv_python.exists() else sys.executable

    cmd = [py_exec, str(script_path)] + args
    logger.info(f"--- Running {stage_display_name}: {script_path.name} ---")

    stage_start_time = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )

        for line in iter(proc.stdout.readline, ''):
            sys.stdout.write(line)
            sys.stdout.flush()
            lower_line = line.lower()
            if "[warning]" in lower_line or "warning:" in lower_line or "status: low-content" in lower_line or "status: connection-error" in lower_line:
                GLOBAL_TRACKER["has_warnings"] = True

        proc.stdout.close()
        return_code = proc.wait()
        elapsed = time.time() - stage_start_time

        if return_code != 0:
            logger.error(f"{stage_display_name} failed with exit code {return_code} ({elapsed:.2f}s).")
            GLOBAL_TRACKER["has_warnings"] = True
            return False, return_code

        logger.info(f"{stage_display_name} completed successfully ({elapsed:.2f}s).")
        return True, 0
    except Exception as e:
        logger.error(f"Exception while executing {stage_display_name}: {e}")
        GLOBAL_TRACKER["has_warnings"] = True
        return False, 1


def execute_stage_with_checkpoint(
    stage_key: str,
    script_path: Path,
    args: List[str],
    display_name: str,
    state_mgr: Optional[Any],
    force: bool = False,
    params: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Checks stage cache/hash state:
      - If current and not force: skips execution and logs '[skip] stage already current'.
      - Otherwise: records start, executes subprocess, records finish.
    """
    if state_mgr:
        is_current, reason = state_mgr.is_stage_current(stage_key, force=force)
        if is_current and not force:
            print(f"[skip] stage already current: {display_name} ({reason})")
            logger.info(f"[skip] {stage_key} already current ({reason})")
            return True

        # Record start timestamp and current input hashes
        state_mgr.record_stage_start(stage_key, parameters=params)

    # Execute the stage
    success, exit_code = run_stage_command(script_path, args, display_name)

    # Record completion status
    if state_mgr:
        state_mgr.record_stage_finish(
            stage_key,
            success=success,
            exit_code=exit_code,
            error_message=None if success else f"Exited with code {exit_code}"
        )

    return success


def run_isolated_stage(
    stage_name_input: str,
    top_n: int = 504,
    clusters_k: int = 10,
    stem: bool = False,
    web_expand: bool = False,
    dashboard: bool = False,
    lang: str = "en",
    force: bool = True,
    min_words: int = 50,
    cluster_seed: Optional[str] = None,
    recency_decay: bool = True,
    decay_factor: float = 0.85,
    momentum_weight: float = 0.15
) -> bool:
    """
    Executes exactly one named stage in isolation using existing outputs/.
    """
    GLOBAL_TRACKER["start_time"] = time.time()
    GLOBAL_TRACKER["has_warnings"] = False

    state_mgr = StateManagerClass(state_file=PATH_RUN_STATE, base_dir=BASE_DIR) if StateManagerClass else None
    
    # Print status table at start of run
    if state_mgr:
        state_mgr.print_status_table()

    canonical_name = resolve_stage_name_func(stage_name_input) if resolve_stage_name_func else stage_name_input
    if not canonical_name:
        available = get_all_stage_names_func() if get_all_stage_names_func else list(STAGE_DEFINITIONS.keys())
        print(f"\n[ERROR] Unknown stage '{stage_name_input}'. Available stages: {', '.join(available)}\n")
        total_elapsed = time.time() - GLOBAL_TRACKER["start_time"]
        print(f"Total elapsed time: {total_elapsed:.2f}s | Verdict: degraded\n")
        return False

    print("\n" + "=" * 70)
    print(f" CENTROID — Running Isolated Stage: {canonical_name}")
    print("=" * 70 + "\n")

    stopwords_lang_file = BASE_DIR / "wordlists" / f"stopwords_{lang}.txt"
    stopwords_default_file = BASE_DIR / "wordlists" / "stopwords_en.txt"
    target_stopwords = stopwords_lang_file if stopwords_lang_file.exists() else stopwords_default_file

    success = False
    if canonical_name == "01_fetch":
        fetch_args = ["--clean"] if force else []
        success = execute_stage_with_checkpoint(
            "01_fetch", SCRIPT_01_FETCH, fetch_args, "Stage 01a (URL Fetch)",
            state_mgr, force=force
        )

    elif canonical_name == "02_extract_pdf":
        success = execute_stage_with_checkpoint(
            "02_extract_pdf", SCRIPT_02_EXTRACT, [], "Stage 01b (PDF & Doc Extraction)",
            state_mgr, force=force
        )

    elif canonical_name == "03_scan_words":
        if not PATH_CORPUS.exists() or PATH_CORPUS.stat().st_size == 0:
            print(f"\n[ERROR] Prerequisite corpus '{PATH_CORPUS}' is missing or empty.")
            print("  Run '01_fetch' and/or '02_extract_pdf' first to ingest data into corpus.txt.\n")
            return False
        scan_args = ["--min-words", str(min_words)]
        if stem:
            scan_args.append("--stem")
        if target_stopwords.exists():
            scan_args.extend(["--stopwords", str(target_stopwords)])
        success = execute_stage_with_checkpoint(
            "03_scan_words", SCRIPT_03_SCAN, scan_args, "Stage 02 (Word Scanner)",
            state_mgr, force=force, params={"stem": stem, "stopwords": str(target_stopwords), "min_words": min_words}
        )

    elif canonical_name == "04_web_expand":
        if not PATH_RAW_CSV.exists() or PATH_RAW_CSV.stat().st_size == 0:
            print(f"\n[ERROR] Prerequisite raw frequencies file '{PATH_RAW_CSV}' is missing or empty.")
            print("  Run '03_scan_words' first to generate word_freq_raw.csv.\n")
            return False
        expand_args = ["--web-expand", "--top", "50"]
        success = execute_stage_with_checkpoint(
            "04_web_expand", SCRIPT_04_EXPAND, expand_args, "Stage 03 (Web Expand)",
            state_mgr, force=force, params={"top": 50}
        )

    elif canonical_name == "05_cluster":
        if not PATH_RAW_CSV.exists() or not PATH_CORPUS.exists():
            print(f"\n[ERROR] Prerequisite raw frequencies '{PATH_RAW_CSV}' or corpus '{PATH_CORPUS}' is missing.")
            print("  Run '03_scan_words' first to generate word_freq_raw.csv.\n")
            return False
        cluster_args = ["--top", str(top_n), "--clusters", str(clusters_k)]
        if cluster_seed:
            cluster_args.extend(["--cluster-seed", str(cluster_seed)])
        success = execute_stage_with_checkpoint(
            "05_cluster", SCRIPT_05_CLUSTER, cluster_args, "Stage 04 (Establish Medoids & Cluster)",
            state_mgr, force=force, params={"top": top_n, "clusters": clusters_k, "cluster_seed": cluster_seed}
        )

    elif canonical_name == "06_report":
        if not PATH_RANKED_CSV.exists() or not PATH_CLUSTERS_JSON.exists():
            print(f"\n[ERROR] Prerequisite ranked CSV '{PATH_RANKED_CSV}' or clusters JSON '{PATH_CLUSTERS_JSON}' is missing.")
            print("  Run '05_cluster' first to generate ranked words and clusters.\n")
            return False
        report_args = ["--top", str(top_n)]
        success = execute_stage_with_checkpoint(
            "06_report", SCRIPT_06_REPORT, report_args, "Stage 05/06 (Report Generator)",
            state_mgr, force=force, params={"top": top_n}
        )

    elif canonical_name == "07_dashboard":
        if not PATH_RANKED_CSV.exists() or not PATH_CLUSTERS_JSON.exists():
            print(f"\n[ERROR] Prerequisite ranked CSV '{PATH_RANKED_CSV}' or clusters JSON '{PATH_CLUSTERS_JSON}' is missing.")
            print("  Run '05_cluster' first to generate ranked words and clusters.\n")
            return False
        tpl_path = BASE_DIR / "outputs" / "dashboard_template.html"
        if not tpl_path.exists():
            print(f"\n[ERROR] Dashboard template '{tpl_path}' is missing.\n")
            return False
        dash_args = ["--top", str(top_n)]
        success = execute_stage_with_checkpoint(
            "07_dashboard", SCRIPT_07_DASHBOARD, dash_args, "Stage 07 (Dashboard Generator)",
            state_mgr, force=force, params={"top": top_n}
        )

    elif canonical_name == "08_drift":
        if not PATH_CLUSTERS_JSON.exists():
            print(f"\n[ERROR] Prerequisite clusters JSON '{PATH_CLUSTERS_JSON}' is missing.")
            print("  Run '05_cluster' first to generate cluster vectors.\n")
            return False
        drift_args = ["--decay-factor", str(decay_factor), "--momentum-weight", str(momentum_weight)]
        if not recency_decay:
            drift_args.append("--no-recency-decay")
        success = execute_stage_with_checkpoint(
            "08_drift", SCRIPT_08_DRIFT, drift_args, "Stage 08 (Super Centroid Delivery & Drift)",
            state_mgr, force=force
        )

    else:
        logger.error(f"Unrecognized canonical stage handler for: {canonical_name}")
        success = False

    total_elapsed = time.time() - GLOBAL_TRACKER["start_time"]
    verdict = "degraded" if (not success or GLOBAL_TRACKER["has_warnings"]) else "healthy"
    print(f"Total elapsed time: {total_elapsed:.2f}s | Verdict: {verdict}\n")
    return success


def run_pipeline(
    top_n: int = 504,
    clusters_k: int = 10,
    stem: bool = False,
    web_expand: bool = False,
    report_only: bool = False,
    dashboard: bool = False,
    lang: str = "en",
    force: bool = False,
    single_stage: Optional[str] = None,
    min_words: int = 50,
    snapshot: Optional[str] = None,
    drift: bool = False,
    cluster_seed: Optional[str] = None,
    recency_decay: bool = True,
    decay_factor: float = 0.85,
    momentum_weight: float = 0.15
) -> bool:
    """
    Main orchestration routine for running the CENTROID pipeline stages with state management.
    """
    GLOBAL_TRACKER["start_time"] = time.time()
    GLOBAL_TRACKER["has_warnings"] = False

    # If a single stage was requested, route to isolated runner
    if single_stage:
        return run_isolated_stage(
            single_stage,
            top_n=top_n,
            clusters_k=clusters_k,
            stem=stem,
            web_expand=web_expand,
            dashboard=dashboard,
            lang=lang,
            force=force,
            min_words=min_words,
            cluster_seed=cluster_seed,
            recency_decay=recency_decay,
            decay_factor=decay_factor,
            momentum_weight=momentum_weight
        )

    state_mgr = StateManagerClass(state_file=PATH_RUN_STATE, base_dir=BASE_DIR) if StateManagerClass else None

    # Print a one-line pipeline status table at the start of every run.py invocation
    if state_mgr:
        state_mgr.print_status_table()

    print("=" * 78)
    print(" CENTROID — Local Word Frequency & Semantic Cluster Analysis Engine")
    print("=" * 78)
    print(f" Config: Top Words={top_n} | Clusters={clusters_k} | Stemming={'ON' if stem else 'OFF'} | "
          f"Web Expand={'ON' if web_expand else 'OFF'} | Report Only={'ON' if report_only else 'OFF'} | "
          f"Dashboard={'ON' if dashboard else 'OFF'} | Drift={'ON' if (drift or dashboard) else 'OFF'} | "
          f"Snapshot={snapshot if snapshot is not None else 'OFF'} | Force={'ON' if force else 'OFF'} | Lang={lang} | Min Words={min_words}")
    print("=" * 78 + "\n")

    # Determine stopwords path based on --lang
    stopwords_lang_file = BASE_DIR / "wordlists" / f"stopwords_{lang}.txt"
    stopwords_default_file = BASE_DIR / "wordlists" / "stopwords_en.txt"
    target_stopwords_path = stopwords_lang_file if stopwords_lang_file.exists() else stopwords_default_file

    # -------------------------------------------------------------
    # REPORT-ONLY MODE
    # -------------------------------------------------------------
    if report_only:
        logger.info("[REPORT-ONLY] Skipping Ingest, Word Scan, and Web Expansion stages.")

        # Ensure prerequisite cluster data and ranked words exist
        if not PATH_RANKED_CSV.exists() or not PATH_CLUSTERS_JSON.exists():
            logger.info("Prerequisite ranked/cluster data missing. Running Stage 04 (Cluster) first...")
            cluster_args = ["--top", str(top_n), "--clusters", str(clusters_k)]
            if not execute_stage_with_checkpoint(
                "05_cluster", SCRIPT_05_CLUSTER, cluster_args, "Stage 04 (Rank + Cluster)",
                state_mgr, force=force, params={"top": top_n, "clusters": clusters_k}
            ):
                logger.error("Failed to generate prerequisite clusters for report.")
                total_elapsed = time.time() - GLOBAL_TRACKER["start_time"]
                print(f"Total elapsed time: {total_elapsed:.2f}s | Verdict: degraded\n")
                return False

        # Run Stage 06: Report Generation
        report_args = ["--top", str(top_n)]
        report_success = execute_stage_with_checkpoint(
            "06_report", SCRIPT_06_REPORT, report_args, "Stage 05/06 (Report Generator)",
            state_mgr, force=force, params={"top": top_n}
        )
        if not report_success:
            total_elapsed = time.time() - GLOBAL_TRACKER["start_time"]
            print(f"Total elapsed time: {total_elapsed:.2f}s | Verdict: degraded\n")
            return False

        # Optional Stage 07: Interactive Dashboard
        if dashboard:
            dash_args = ["--top", str(top_n)]
            execute_stage_with_checkpoint(
                "07_dashboard", SCRIPT_07_DASHBOARD, dash_args, "Stage 07 (Dashboard Generator)",
                state_mgr, force=force, params={"top": top_n}
            )

        total_elapsed = time.time() - GLOBAL_TRACKER["start_time"]
        verdict = "degraded" if GLOBAL_TRACKER["has_warnings"] else "healthy"

        print("\n" + "=" * 78)
        print(" CENTROID Pipeline Finished Successfully (Report-Only Mode)!")
        print(f" Report Generated:   {PATH_REPORT_MD.resolve()}")
        if dashboard and PATH_DASHBOARD_HTML.exists():
            print(f" Dashboard Location: {PATH_DASHBOARD_HTML.resolve()}")
            print(f" Standalone URL:     file:///{str(PATH_DASHBOARD_HTML.resolve()).replace(chr(92), '/')}")
        print("=" * 78)
        print(f"Total elapsed time: {total_elapsed:.2f}s | Verdict: {verdict}")
        print("=" * 78 + "\n")
        return True

    # -------------------------------------------------------------
    # STAGE 01: INGEST (01_fetch.py & 02_extract_pdf.py)
    # -------------------------------------------------------------
    logger.info(">>> STAGE 01: INGESTION (URLs, PDFs, Docs, Raw Text)")

    # 1a. Fetch URLs
    if SCRIPT_01_FETCH.exists():
        fetch_args = ["--clean"]
        execute_stage_with_checkpoint(
            "01_fetch", SCRIPT_01_FETCH, fetch_args, "Stage 01a (URL Fetch)",
            state_mgr, force=force
        )
    else:
        logger.debug("01_fetch.py not found; skipping URL fetch.")

    # 1b. Extract PDFs and Docs
    if SCRIPT_02_EXTRACT.exists():
        execute_stage_with_checkpoint(
            "02_extract_pdf", SCRIPT_02_EXTRACT, [], "Stage 01b (PDF & Doc Extraction)",
            state_mgr, force=force
        )
    else:
        logger.debug("02_extract_pdf.py not found; skipping PDF/doc extraction.")

    if not PATH_CORPUS.exists() or PATH_CORPUS.stat().st_size == 0:
        logger.warning(f"Corpus file '{PATH_CORPUS}' is empty or does not exist. Proceeding with available data.")

    # -------------------------------------------------------------
    # STAGE 02: WORD SCANNER (03_scan_words.py)
    # -------------------------------------------------------------
    logger.info("\n>>> STAGE 02: WORD SCANNER (Tokenize, Filter, Compute TF-IDF)")
    scan_args: List[str] = ["--top", str(top_n), "--min-words", str(min_words)]
    if stem:
        scan_args.append("--stem")
    if target_stopwords_path.exists():
        scan_args.extend(["--stopwords", str(target_stopwords_path)])

    scan_params = {"top": top_n, "stem": stem, "stopwords": str(target_stopwords_path), "min_words": min_words}
    if not execute_stage_with_checkpoint(
        "03_scan_words", SCRIPT_03_SCAN, scan_args, "Stage 02 (Word Scanner)",
        state_mgr, force=force, params=scan_params
    ):
        logger.error("Stage 02 (Word Scanner) failed. Halting pipeline.")
        total_elapsed = time.time() - GLOBAL_TRACKER["start_time"]
        print("=" * 78)
        print(f"Total elapsed time: {total_elapsed:.2f}s | Verdict: degraded")
        print("=" * 78 + "\n")
        return False

    # -------------------------------------------------------------
    # STAGE 03: WEB EXPANSION (Optional: 04_web_expand.py)
    # -------------------------------------------------------------
    if web_expand:
        logger.info("\n>>> STAGE 03: WEB EXPANSION (DuckDuckGo Keyword Expansion)")
        expand_args = ["--web-expand", "--top", "50"]
        expand_params = {"top": 50}
        if execute_stage_with_checkpoint(
            "04_web_expand", SCRIPT_04_EXPAND, expand_args, "Stage 03 (Web Expand)",
            state_mgr, force=force, params=expand_params
        ):
            # If web expand added new vocabulary, re-run Stage 02 Word Scanner with force=True
            logger.info("Re-indexing Stage 02 Word Scanner for newly expanded web vocabulary...")
            execute_stage_with_checkpoint(
                "03_scan_words", SCRIPT_03_SCAN, scan_args, "Stage 02 (Word Scanner - Post Expansion)",
                state_mgr, force=True, params=scan_params
            )
        else:
            logger.warning("Stage 03 (Web Expand) encountered an issue; continuing with existing corpus.")
    else:
        if state_mgr:
            state_mgr.record_stage_skipped("04_web_expand", reason="Web expansion not enabled (--web-expand flag not passed)")

    # -------------------------------------------------------------
    # STAGE 04: ESTABLISH MEDOIDS & CLUSTER (05_cluster.py)
    # -------------------------------------------------------------
    logger.info("\n>>> STAGE 04: ESTABLISH MEDOIDS & CLUSTER (Word2Vec + Plan B Pairwise Dissimilarity)")
    cluster_args = [
        "--top", str(top_n),
        "--clusters", str(clusters_k)
    ]
    if cluster_seed:
        cluster_args.extend(["--cluster-seed", str(cluster_seed)])
    if snapshot is not None:
        if snapshot:
            cluster_args.extend(["--snapshot", str(snapshot)])
        else:
            cluster_args.append("--snapshot")
    cluster_params = {"top": top_n, "clusters": clusters_k, "snapshot": snapshot, "cluster_seed": cluster_seed}
    if not execute_stage_with_checkpoint(
        "05_cluster", SCRIPT_05_CLUSTER, cluster_args, "Stage 04 (Establish Medoids & Cluster)",
        state_mgr, force=force, params=cluster_params
    ):
        logger.error("Stage 04 (Establish Medoids & Cluster) failed. Halting pipeline.")
        total_elapsed = time.time() - GLOBAL_TRACKER["start_time"]
        print("=" * 78)
        print(f"Total elapsed time: {total_elapsed:.2f}s | Verdict: degraded")
        print("=" * 78 + "\n")
        return False

    # -------------------------------------------------------------
    # STAGE 05/06: REPORT GENERATOR (06_report.py)
    # -------------------------------------------------------------
    logger.info("\n>>> STAGE 05/06: REPORT GENERATION (Markdown Summary)")
    report_args = ["--top", str(top_n)]
    report_params = {"top": top_n}
    if not execute_stage_with_checkpoint(
        "06_report", SCRIPT_06_REPORT, report_args, "Stage 05/06 (Report Generator)",
        state_mgr, force=force, params=report_params
    ):
        logger.error("Stage 06 (Report Generator) failed. Halting pipeline.")
        total_elapsed = time.time() - GLOBAL_TRACKER["start_time"]
        print("=" * 78)
        print(f"Total elapsed time: {total_elapsed:.2f}s | Verdict: degraded")
        print("=" * 78 + "\n")
        return False

    # -------------------------------------------------------------
    # STAGE 08: SUPER CENTROID DELIVERY & TRAJECTORY KINEMATICS (08_drift.py)
    # -------------------------------------------------------------
    if drift or dashboard:
        logger.info("\n>>> STAGE 08: SUPER CENTROID DELIVERY & TRAJECTORY KINEMATICS")
        drift_args = ["--decay-factor", str(decay_factor), "--momentum-weight", str(momentum_weight)]
        if not recency_decay:
            drift_args.append("--no-recency-decay")
        execute_stage_with_checkpoint(
            "08_drift", SCRIPT_08_DRIFT, drift_args, "Stage 08 (Super Centroid Delivery & Drift)",
            state_mgr, force=force
        )
    else:
        if state_mgr:
            state_mgr.record_stage_skipped("08_drift", reason="Drift / Super Centroid analysis not requested (--drift flag not passed)")

    # -------------------------------------------------------------
    # STAGE 07: INTERACTIVE DASHBOARD (Optional: 07_dashboard.py)
    # -------------------------------------------------------------
    if dashboard:
        logger.info("\n>>> STAGE 07: INTERACTIVE DASHBOARD GENERATOR")
        dash_args = ["--top", str(top_n)]
        dash_params = {"top": top_n}
        execute_stage_with_checkpoint(
            "07_dashboard", SCRIPT_07_DASHBOARD, dash_args, "Stage 07 (Dashboard Generator)",
            state_mgr, force=force, params=dash_params
        )
    else:
        if state_mgr:
            state_mgr.record_stage_skipped("07_dashboard", reason="Dashboard not requested (--dashboard flag not passed)")

    # -------------------------------------------------------------
    # COMPLETION
    # -------------------------------------------------------------
    total_elapsed = time.time() - GLOBAL_TRACKER["start_time"]
    verdict = "degraded" if GLOBAL_TRACKER["has_warnings"] else "healthy"

    print("\n" + "=" * 78)
    print(" CENTROID Pipeline Completed Successfully!")
    print(f" State File:     {PATH_RUN_STATE.resolve()}")
    print(f" Final Report:   {PATH_REPORT_MD.resolve()}")
    if dashboard and PATH_DASHBOARD_HTML.exists():
        print(f" Dashboard HTML: {PATH_DASHBOARD_HTML.resolve()}")
        print(f" Standalone URL: file:///{str(PATH_DASHBOARD_HTML.resolve()).replace(chr(92), '/')}")
    print(f" Ranked CSV:     {PATH_RANKED_CSV.resolve()}")
    print(f" Clusters JSON:  {PATH_CLUSTERS_JSON.resolve()}")
    print(f" Raw Freq CSV:   {PATH_RAW_CSV.resolve()}")
    print("=" * 78)
    print(f"Total elapsed time: {total_elapsed:.2f}s | Verdict: {verdict}")
    print("=" * 78 + "\n")
    return True


def parse_arguments() -> argparse.Namespace:
    """
    Parses CLI flags conforming to AGENT.md Section 7 specification and stage state tracking.
    """
    parser = argparse.ArgumentParser(
        description="CENTROID — Local word-frequency and semantic cluster analysis engine."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=1000,
        help="Number of top words in report/dashboard (default: 1000)."
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=10,
        help="Number of KMeans clusters (default: 10)."
    )
    parser.add_argument(
        "--stem",
        action="store_true",
        default=False,
        help="Enable Porter stemmer for word tokenization."
    )
    parser.add_argument(
        "--web-expand",
        action="store_true",
        default=False,
        help="Enable DuckDuckGo web search expansion on top words."
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        default=False,
        help="Skip ingest and scan stages; regenerate report only."
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=False,
        help="Generate interactive HTML dashboard (outputs/dashboard.html)."
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code for stopwords (default: 'en')."
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        default=False,
        help="Force execution of stages even if inputs are unchanged and stage succeeded."
    )
    parser.add_argument(
        "--stage",
        "-s",
        type=str,
        default=None,
        help="Execute exactly one named stage in isolation (e.g. fetch, extract, scan, expand, cluster, report, dashboard)."
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=50,
        help="Minimum corpus word count threshold for scanning (default: 50)."
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Archive current cluster run as an indexed temporal snapshot in outputs/snapshots/."
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        default=False,
        help="Run cross-temporal Orthogonal Procrustes semantic drift analysis and trajectory forecast."
    )
    parser.add_argument(
        "--cluster-seed", "--target-cluster",
        dest="cluster_seed",
        type=str,
        default=None,
        help="Specific cluster focus or target word seed to study (e.g. 'deepfakes')."
    )
    parser.add_argument(
        "--recency-decay",
        dest="recency_decay",
        action="store_true",
        default=True,
        help="Enable exponential recency decay weighting for historical medoids (default: True)."
    )
    parser.add_argument(
        "--no-recency-decay",
        dest="recency_decay",
        action="store_false",
        help="Disable recency decay and use uniform averaging across snapshots."
    )
    parser.add_argument(
        "--decay-factor",
        type=float,
        default=0.85,
        help="Decay factor gamma for recency weighting (default: 0.85)."
    )
    parser.add_argument(
        "--momentum-weight",
        type=float,
        default=0.15,
        help="Momentum factor lambda for velocity vector in Super Centroid synthesis (default: 0.15)."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the interactive CENTROID GUI server and open dashboard.html in your web browser."
    )
    parser.add_argument(
        "--gui-port",
        type=int,
        default=8000,
        help="Local port binding for the GUI server (default: 8000)."
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically when starting the GUI server."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.gui:
        try:
            from pipeline.gui_server import start_gui_server
        except ImportError:
            from gui_server import start_gui_server
        try:
            start_gui_server(port=args.gui_port, open_browser=not args.no_browser)
            sys.exit(0)
        except KeyboardInterrupt:
            print("\n[CENTROID] GUI server stopped by user.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"GUI server error: {e}", exc_info=True)
            sys.exit(1)

    try:
        success = run_pipeline(
            top_n=args.top,
            clusters_k=args.clusters,
            stem=args.stem,
            web_expand=args.web_expand,
            report_only=args.report_only,
            dashboard=args.dashboard,
            lang=args.lang,
            force=args.force,
            single_stage=args.stage,
            min_words=args.min_words,
            snapshot=args.snapshot,
            drift=args.drift,
            cluster_seed=args.cluster_seed,
            recency_decay=args.recency_decay,
            decay_factor=args.decay_factor,
            momentum_weight=args.momentum_weight
        )
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[CENTROID] Execution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
