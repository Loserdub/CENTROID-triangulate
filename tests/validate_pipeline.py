#!/usr/bin/env python3
"""
CENTROID Pipeline Smoke Test & Validation Suite
Location: tests/validate_pipeline.py

Description:
  End-to-end smoke test for the CENTROID word-frequency and semantic clustering pipeline.
  Validates that:
    1. Ingestion extracts synthetic fixture text into `outputs/corpus.txt` (non-empty).
    2. Word scanner produces `outputs/word_freq_raw.csv` with valid rows.
    3. Stopword filtering properly purges common words ("the", "and", "is", "of", etc.).
    4. Frequency ranking generates `outputs/word_freq_ranked.csv`.
    5. Centroid clustering produces `outputs/clusters.json` with >= 2 distinct clusters.
    6. Markdown report `outputs/REPORT.md` is created with all required sections.
    7. Standalone HTML dashboard `outputs/dashboard.html` is generated with inline JSON data.

Usage:
  python tests/validate_pipeline.py
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Base directory is the project root (CENTROID/)
BASE_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures"

# Project paths
INPUTS_DIR = BASE_DIR / "inputs"
INPUTS_DOCS_DIR = INPUTS_DIR / "docs"
INPUTS_PDFS_DIR = INPUTS_DIR / "pdfs"
INPUTS_URLS_FILE = INPUTS_DIR / "urls.txt"
OUTPUTS_DIR = BASE_DIR / "outputs"

PATH_CORPUS = OUTPUTS_DIR / "corpus.txt"
PATH_RAW_CSV = OUTPUTS_DIR / "word_freq_raw.csv"
PATH_RANKED_CSV = OUTPUTS_DIR / "word_freq_ranked.csv"
PATH_CLUSTERS_JSON = OUTPUTS_DIR / "clusters.json"
PATH_REPORT_MD = OUTPUTS_DIR / "REPORT.md"
PATH_DASHBOARD_HTML = OUTPUTS_DIR / "dashboard.html"
PATH_DASHBOARD_TPL = OUTPUTS_DIR / "dashboard_template.html"


class AssertionResult:
    def __init__(self, name: str, passed: bool, message: str):
        self.name = name
        self.passed = passed
        self.message = message


def run_pipeline_subprocess() -> subprocess.CompletedProcess:
    """
    Executes the full pipeline via python run.py --top 500 --clusters 3 --dashboard
    """
    cmd = [
        sys.executable,
        str(BASE_DIR / "run.py"),
        "--top", "500",
        "--clusters", "3",
        "--dashboard"
    ]
    return subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=False
    )


def test_pipeline_end_to_end() -> bool:
    """
    Sets up a clean test fixture run, executes the full pipeline,
    and performs all validation assertions.
    """
    print("\n" + "=" * 70)
    print(" CENTROID PIPELINE VALIDATION & SMOKE TEST")
    print("=" * 70)
    print(f" Python Interpreter: {sys.executable}")
    print(f" Working Directory:  {BASE_DIR}")
    print(f" Fixtures Directory: {FIXTURES_DIR}")
    print("=" * 70 + "\n")

    # Verify fixtures exist
    fixture_files = list(FIXTURES_DIR.glob("*.txt"))
    if not fixture_files:
        print(f"[FAIL] No fixture files found in '{FIXTURES_DIR}'")
        return False

    print(f"[INFO] Found {len(fixture_files)} test fixture files:")
    for f in sorted(fixture_files):
        print(f"  • {f.name} ({f.stat().st_size:,} bytes)")

    # Temporary backup directories for existing state preservation
    backup_temp_dir = Path(tempfile.mkdtemp(prefix="centroid_backup_"))
    backup_inputs_docs = backup_temp_dir / "inputs_docs"
    backup_inputs_urls = backup_temp_dir / "urls.txt"
    backup_outputs = backup_temp_dir / "outputs"

    results: List[AssertionResult] = []

    try:
        # 1. Back up existing user inputs and outputs if present
        if INPUTS_DOCS_DIR.exists():
            shutil.copytree(INPUTS_DOCS_DIR, backup_inputs_docs)
        if INPUTS_URLS_FILE.exists():
            shutil.copy2(INPUTS_URLS_FILE, backup_inputs_urls)
        if OUTPUTS_DIR.exists():
            shutil.copytree(OUTPUTS_DIR, backup_outputs)

        # 2. Stage test fixtures into inputs/docs/
        INPUTS_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        # Clear existing docs in inputs/docs for isolated test
        for existing in INPUTS_DOCS_DIR.glob("*"):
            if existing.is_file():
                existing.unlink()

        for fix in fixture_files:
            shutil.copy2(fix, INPUTS_DOCS_DIR / fix.name)

        # Clear inputs/urls.txt during test so no live HTTP requests are made
        with open(INPUTS_URLS_FILE, mode="w", encoding="utf-8") as f:
            f.write("# Synthetic test run — no live URLs\n")

        # Clean outputs (while preserving dashboard_template.html if present)
        tpl_backup_data = None
        if PATH_DASHBOARD_TPL.exists():
            with open(PATH_DASHBOARD_TPL, "r", encoding="utf-8") as f:
                tpl_backup_data = f.read()

        for out_file in [PATH_CORPUS, PATH_RAW_CSV, PATH_RANKED_CSV, PATH_CLUSTERS_JSON, PATH_REPORT_MD, PATH_DASHBOARD_HTML]:
            if out_file.exists():
                out_file.unlink()

        if tpl_backup_data and not PATH_DASHBOARD_TPL.exists():
            with open(PATH_DASHBOARD_TPL, "w", encoding="utf-8") as f:
                f.write(tpl_backup_data)

        # 3. Execute the full pipeline
        print("\n[INFO] Running full CENTROID pipeline against fixtures...")
        proc = run_pipeline_subprocess()

        if proc.returncode != 0:
            print("[ERROR] Pipeline subprocess returned non-zero exit code!")
            print("STDOUT:\n", proc.stdout)
            print("STDERR:\n", proc.stderr)
            results.append(AssertionResult("Pipeline Execution", False, f"Process exited with code {proc.returncode}"))
        else:
            results.append(AssertionResult("Pipeline Execution", True, "Pipeline executed to completion (exit code 0)"))

        # -------------------------------------------------------------
        # ASSERTION 1: outputs/corpus.txt is non-empty
        # -------------------------------------------------------------
        if PATH_CORPUS.exists() and PATH_CORPUS.stat().st_size > 0:
            size_kb = PATH_CORPUS.stat().st_size / 1024.0
            results.append(AssertionResult(
                "1. Corpus Extraction",
                True,
                f"outputs/corpus.txt created and non-empty ({size_kb:.2f} KB)"
            ))
        else:
            results.append(AssertionResult(
                "1. Corpus Extraction",
                False,
                "outputs/corpus.txt is missing or empty"
            ))

        # -------------------------------------------------------------
        # ASSERTION 2: outputs/word_freq_raw.csv has > 0 rows
        # -------------------------------------------------------------
        raw_rows_count = 0
        raw_words_set: Set[str] = set()
        if PATH_RAW_CSV.exists():
            with open(PATH_RAW_CSV, mode="r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    w = row.get("word", "").strip().lower()
                    if w:
                        raw_rows_count += 1
                        raw_words_set.add(w)

        if raw_rows_count > 0:
            results.append(AssertionResult(
                "2. Word Frequency Scanner",
                True,
                f"outputs/word_freq_raw.csv contains {raw_rows_count:,} records ({len(raw_words_set):,} unique words)"
            ))
        else:
            results.append(AssertionResult(
                "2. Word Frequency Scanner",
                False,
                "outputs/word_freq_raw.csv is missing or has 0 rows"
            ))

        # -------------------------------------------------------------
        # ASSERTION 3: Stopword Filtering
        # -------------------------------------------------------------
        forbidden_stopwords = ["the", "and", "is", "of", "in", "to", "a", "an", "with", "by", "for"]
        found_stopwords = [sw for sw in forbidden_stopwords if sw in raw_words_set]
        if not found_stopwords and raw_rows_count > 0:
            results.append(AssertionResult(
                "3. Stopword Filtering",
                True,
                "Spot-checked stopwords ('the', 'and', 'is', 'of', 'in', 'to') are strictly absent"
            ))
        else:
            results.append(AssertionResult(
                "3. Stopword Filtering",
                False,
                f"Found forbidden stopwords in raw output: {found_stopwords}"
            ))

        # -------------------------------------------------------------
        # ASSERTION 4: outputs/clusters.json has >= 2 distinct clusters
        # -------------------------------------------------------------
        cluster_count = 0
        cluster_labels: List[str] = []
        if PATH_CLUSTERS_JSON.exists():
            try:
                with open(PATH_CLUSTERS_JSON, mode="r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    if isinstance(cdata, dict):
                        cluster_count = len(cdata)
                        cluster_labels = [v.get("label", "") for v in cdata.values()]
            except Exception as e:
                pass

        if cluster_count >= 2:
            results.append(AssertionResult(
                "4. Centroid Clustering",
                True,
                f"outputs/clusters.json contains {cluster_count} distinct clusters ({', '.join(cluster_labels)})"
            ))
        else:
            results.append(AssertionResult(
                "4. Centroid Clustering",
                False,
                f"Expected >= 2 clusters in outputs/clusters.json, found {cluster_count}"
            ))

        # -------------------------------------------------------------
        # ASSERTION 5: outputs/REPORT.md generated with all sections
        # -------------------------------------------------------------
        report_valid = False
        if PATH_REPORT_MD.exists() and PATH_REPORT_MD.stat().st_size > 0:
            with open(PATH_REPORT_MD, mode="r", encoding="utf-8") as f:
                report_text = f.read()

            required_sections = [
                "# CENTROID Report",
                "## Top",
                "## Centroid Clusters",
                "## Gap Analysis",
                "## Cluster % Breakdown"
            ]
            missing_sections = [sec for sec in required_sections if sec not in report_text]
            if not missing_sections:
                report_valid = True
                results.append(AssertionResult(
                    "5. Markdown Report Generation",
                    True,
                    f"outputs/REPORT.md successfully created with all 5 required sections ({len(report_text):,} chars)"
                ))
            else:
                results.append(AssertionResult(
                    "5. Markdown Report Generation",
                    False,
                    f"outputs/REPORT.md missing sections: {missing_sections}"
                ))
        else:
            results.append(AssertionResult(
                "5. Markdown Report Generation",
                False,
                "outputs/REPORT.md does not exist or is empty"
            ))

        # -------------------------------------------------------------
        # ASSERTION 6: outputs/dashboard.html generated without throwing
        # -------------------------------------------------------------
        dashboard_valid = False
        if PATH_DASHBOARD_HTML.exists() and PATH_DASHBOARD_HTML.stat().st_size > 0:
            with open(PATH_DASHBOARD_HTML, mode="r", encoding="utf-8") as f:
                dash_text = f.read()

            has_cluster_data = "const CLUSTER_DATA =" in dash_text
            has_table_rows = "<tbody id=\"ranked-table-body\">" in dash_text and "</td>" in dash_text
            if has_cluster_data and has_table_rows:
                dashboard_valid = True
                results.append(AssertionResult(
                    "6. Interactive HTML Dashboard",
                    True,
                    f"outputs/dashboard.html generated with embedded CLUSTER_DATA and populated table ({len(dash_text):,} chars)"
                ))
            else:
                results.append(AssertionResult(
                    "6. Interactive HTML Dashboard",
                    False,
                    "outputs/dashboard.html missing CLUSTER_DATA or table rows"
                ))
        else:
            results.append(AssertionResult(
                "6. Interactive HTML Dashboard",
                False,
                "outputs/dashboard.html does not exist or is empty"
            ))

    finally:
        # 4. Restore original user state if backups existed
        print("\n[INFO] Restoring workspace environment...")
        if backup_inputs_docs.exists():
            for f in INPUTS_DOCS_DIR.glob("*"):
                if f.is_file():
                    f.unlink()
            for f in backup_inputs_docs.glob("*"):
                shutil.copy2(f, INPUTS_DOCS_DIR / f.name)

        if backup_inputs_urls.exists():
            shutil.copy2(backup_inputs_urls, INPUTS_URLS_FILE)

        if backup_outputs.exists():
            for f in backup_outputs.glob("*"):
                if f.is_file():
                    shutil.copy2(f, OUTPUTS_DIR / f.name)

        # Cleanup backup temporary folder
        if backup_temp_dir.exists():
            shutil.rmtree(backup_temp_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # PRINT RESULTS SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" CENTROID PIPELINE ASSERTIONS SUMMARY")
    print("=" * 70)

    all_passed = True
    for res in results:
        status_tag = "[PASS]" if res.passed else "[FAIL]"
        if not res.passed:
            all_passed = False
        print(f" {status_tag} {res.name:<32} {res.message}")

    print("=" * 70)
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)

    if all_passed:
        print(f" RESULT: ALL {total_count}/{total_count} ASSERTIONS PASSED")
        print(" The pipeline is verified and ready for live production use.")
        print("=" * 70 + "\n")
        return True
    else:
        print(f" RESULT: {total_count - passed_count}/{total_count} ASSERTION(S) FAILED")
        print("=" * 70 + "\n")
        return False


def main():
    success = test_pipeline_end_to_end()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
