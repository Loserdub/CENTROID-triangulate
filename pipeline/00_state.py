#!/usr/bin/env python3
"""
CENTROID Pipeline - Stage State & Checkpoint Manager
Specification: pipeline/00_state.py

Description:
  Tracks execution state, start/end timestamps, input file hashes, and exit statuses
  for all CENTROID pipeline stages in `outputs/run_state.json`.

  Enables:
    1. Incremental runs: skips stages whose inputs have not changed and previously completed.
    2. Resumable pipelines: partial failure records failed stage so rerun resumes from the failure point.
    3. Isolated stage execution: supports running a single named stage with `--stage NAME`.
    4. Telemetry: outputs a clean status table at the start of every run.

Usage:
  python pipeline/00_state.py --status
  python pipeline/00_state.py --reset
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Base directory is the project root (CENTROID/)
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DEFAULT_STATE_FILE = OUTPUTS_DIR / "run_state.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("00_state")

# Standard Pipeline Stage Definitions
STAGE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "01_fetch": {
        "script": "01_fetch.py",
        "display_name": "Stage 01a (URL Fetch)",
        "inputs": [
            "inputs/urls.txt",
            "pipeline/01_fetch.py"
        ],
        "outputs": ["outputs/corpus.txt"],
        "aliases": ["01", "01_fetch", "fetch", "urls", "fetch_urls", "ingest_urls"]
    },
    "02_extract_pdf": {
        "script": "02_extract_pdf.py",
        "display_name": "Stage 01b (PDF & Doc Extraction)",
        "inputs": [
            "inputs/pdfs",
            "inputs/docs",
            "inputs/raw",
            "pipeline/02_extract_pdf.py"
        ],
        "outputs": ["outputs/corpus.txt"],
        "aliases": ["02", "02_extract_pdf", "extract", "extract_pdf", "pdf", "docs", "ingest_docs"]
    },
    "03_scan_words": {
        "script": "03_scan_words.py",
        "display_name": "Stage 02 (Word Scanner)",
        "inputs": [
            "outputs/corpus.txt",
            "wordlists/stopwords_en.txt",
            "wordlists/custom_ignore.txt",
            "wordlists/web_coding_lingo.txt",
            "wordlists/target_words.txt",
            "pipeline/03_scan_words.py"
        ],
        "outputs": ["outputs/word_freq_raw.csv"],
        "aliases": ["03", "03_scan_words", "scan", "scan_words", "scanner", "word_scan"]
    },
    "04_web_expand": {
        "script": "04_web_expand.py",
        "display_name": "Stage 03 (Web Expand)",
        "inputs": [
            "outputs/word_freq_raw.csv",
            "pipeline/04_web_expand.py"
        ],
        "outputs": ["outputs/corpus.txt"],
        "aliases": ["04", "04_web_expand", "web_expand", "expand", "web"]
    },
    "05_cluster": {
        "script": "05_cluster.py",
        "display_name": "Stage 04 (Rank + Cluster)",
        "inputs": [
            "outputs/word_freq_raw.csv",
            "outputs/corpus.txt",
            "pipeline/05_cluster.py"
        ],
        "outputs": [
            "outputs/word_freq_ranked.csv",
            "outputs/clusters.json"
        ],
        "aliases": ["05", "05_cluster", "cluster", "clustering", "kmeans", "rank_cluster"]
    },
    "06_report": {
        "script": "06_report.py",
        "display_name": "Stage 05/06 (Report Generator)",
        "inputs": [
            "outputs/word_freq_ranked.csv",
            "outputs/clusters.json",
            "outputs/word_freq_raw.csv",
            "wordlists/target_words.txt",
            "pipeline/06_report.py"
        ],
        "outputs": ["outputs/REPORT.md"],
        "aliases": ["06", "06_report", "report", "report_generator", "md_report"]
    },
    "07_dashboard": {
        "script": "07_dashboard.py",
        "display_name": "Stage 07 (Dashboard Generator)",
        "inputs": [
            "outputs/word_freq_ranked.csv",
            "outputs/clusters.json",
            "outputs/dashboard_template.html",
            "pipeline/07_dashboard.py"
        ],
        "outputs": ["outputs/dashboard.html"],
        "aliases": ["07", "07_dashboard", "dashboard", "html_dashboard", "viz"]
    },
    "08_drift": {
        "script": "08_drift.py",
        "display_name": "Stage 08 (Semantic Drift Tracking)",
        "inputs": [
            "outputs/clusters.json",
            "outputs/timeline_index.json",
            "pipeline/08_drift.py"
        ],
        "outputs": ["outputs/drift_analysis.json"],
        "aliases": ["08", "08_drift", "drift", "procrustes", "trajectory", "forecast"]
    }
}


def compute_file_hash(file_path: Path) -> str:
    """Computes SHA-256 hash of a single file."""
    if not file_path.exists() or not file_path.is_file():
        return "MISSING"
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.debug(f"Error hashing file {file_path}: {e}")
        return "ERROR"


def compute_directory_hash(dir_path: Path) -> str:
    """Computes a deterministic combined SHA-256 hash for a directory's contents."""
    if not dir_path.exists() or not dir_path.is_dir():
        return "EMPTY"
    
    files = sorted([p for p in dir_path.rglob("*") if p.is_file()])
    if not files:
        return "EMPTY"

    h = hashlib.sha256()
    for p in files:
        try:
            rel = str(p.relative_to(dir_path)).replace("\\", "/")
            h.update(rel.encode("utf-8"))
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
        except Exception as e:
            logger.debug(f"Error hashing directory item {p}: {e}")
    return h.hexdigest()


def compute_path_hash(path: Path) -> str:
    """Computes hash for either a file or directory."""
    if not path.exists():
        return "MISSING"
    if path.is_dir():
        return compute_directory_hash(path)
    return compute_file_hash(path)


def compute_stage_input_hashes(stage_name: str, base_dir: Path = BASE_DIR) -> Dict[str, str]:
    """Computes input hashes for all inputs associated with a stage."""
    spec = STAGE_DEFINITIONS.get(stage_name, {})
    inputs = spec.get("inputs", [])
    hashes: Dict[str, str] = {}
    for item in inputs:
        full_path = base_dir / item
        hashes[item] = compute_path_hash(full_path)
    return hashes


def compute_composite_hash(input_hashes: Dict[str, str]) -> str:
    """Computes a composite SHA-256 hash from a dictionary of input hashes."""
    h = hashlib.sha256()
    for k in sorted(input_hashes.keys()):
        h.update(f"{k}:{input_hashes[k]}\n".encode("utf-8"))
    return h.hexdigest()


def resolve_stage_name(name_or_alias: str) -> Optional[str]:
    """Resolves a user-provided stage name or alias to the canonical stage identifier."""
    cleaned = name_or_alias.strip().lower().replace("-", "_")
    for canonical_name, spec in STAGE_DEFINITIONS.items():
        if cleaned == canonical_name.lower():
            return canonical_name
        if cleaned in [alias.lower() for alias in spec.get("aliases", [])]:
            return canonical_name
        if spec.get("script") and cleaned == Path(spec["script"]).stem.lower():
            return canonical_name
    return None


def get_all_stage_names() -> List[str]:
    """Returns a list of all canonical stage names in order."""
    return list(STAGE_DEFINITIONS.keys())


class StateManager:
    """
    Manages reading, updating, and persisting CENTROID pipeline execution state.
    """

    def __init__(self, state_file: Optional[Path] = None, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or BASE_DIR
        self.state_file = state_file or (self.base_dir / "outputs" / "run_state.json")
        self._state: Dict[str, Any] = self._load()

    def _default_state(self) -> Dict[str, Any]:
        """Generates a fresh blank state dictionary."""
        stages = {}
        for name, spec in STAGE_DEFINITIONS.items():
            stages[name] = {
                "name": name,
                "display_name": spec.get("display_name", name),
                "script": spec.get("script", ""),
                "status": "pending",  # pending | running | completed | failed | skipped
                "start_time": None,
                "end_time": None,
                "duration_seconds": None,
                "exit_code": None,
                "error_message": None,
                "input_hashes": {},
                "composite_input_hash": None,
                "parameters": {}
            }
        return {
            "version": "1.0",
            "last_updated": datetime.datetime.now().isoformat(),
            "pipeline_status": "idle",
            "stages": stages
        }

    def _load(self) -> Dict[str, Any]:
        """Loads state from run_state.json or initializes default state."""
        if not self.state_file.exists():
            return self._default_state()

        try:
            with open(self.state_file, mode="r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict) or "stages" not in data:
                    logger.warning("Corrupted run_state.json detected. Reinitializing.")
                    return self._default_state()

                # Ensure all defined stages exist in loaded state
                default = self._default_state()
                for name, default_stage in default["stages"].items():
                    if name not in data["stages"]:
                        data["stages"][name] = default_stage
                return data
        except Exception as e:
            logger.warning(f"Failed to read run_state.json: {e}. Reinitializing.")
            return self._default_state()

    def save(self) -> None:
        """Persists the current state safely to run_state.json."""
        self._state["last_updated"] = datetime.datetime.now().isoformat()
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            # Write atomically using a temporary file
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, mode="w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
            temp_file.replace(self.state_file)
        except Exception as e:
            logger.error(f"Failed to save state to {self.state_file}: {e}")

    def get_stage_state(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Retrieves state dictionary for a specific stage."""
        return self._state.get("stages", {}).get(stage_name)

    def is_stage_current(
        self,
        stage_name: str,
        force: bool = False,
        current_hashes: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, str]:
        """
        Checks whether a stage can be skipped because it already succeeded
        and its input files have not changed.

        Returns:
          (True, reason) if stage is current and can be skipped.
          (False, reason) if stage needs to run.
        """
        if force:
            return False, "Force flag enabled (--force)"

        stage = self.get_stage_state(stage_name)
        if not stage:
            return False, f"Stage {stage_name} has no recorded state"

        if stage.get("status") != "completed":
            prev_status = stage.get("status", "pending")
            return False, f"Previous run status is '{prev_status}'"

        if stage.get("exit_code") != 0:
            return False, f"Previous exit code was {stage.get('exit_code')}"

        # Check that all expected output files exist and are non-empty
        spec = STAGE_DEFINITIONS.get(stage_name, {})
        expected_outputs = spec.get("outputs", [])
        for out_rel_path in expected_outputs:
            out_file = self.base_dir / out_rel_path
            if not out_file.exists():
                return False, f"Required output file '{out_rel_path}' is missing"
            if out_file.is_file() and out_file.stat().st_size == 0:
                return False, f"Required output file '{out_rel_path}' is empty (0 bytes)"

        prev_hashes = stage.get("input_hashes", {})
        if not prev_hashes:
            return False, "No previous input hashes recorded"

        curr_hashes = current_hashes if current_hashes is not None else compute_stage_input_hashes(stage_name, self.base_dir)

        # Check if any input file hash has changed
        for input_key, prev_h in prev_hashes.items():
            curr_h = curr_hashes.get(input_key)
            if curr_h != prev_h:
                return False, f"Input '{input_key}' changed ({prev_h[:8]}... -> {curr_h[:8] if curr_h else 'None'}...)"

        # Check for newly added inputs
        for input_key in curr_hashes.keys():
            if input_key not in prev_hashes:
                return False, f"New input detected: '{input_key}'"

        return True, "Inputs match and previous run completed successfully"

    def record_stage_start(
        self,
        stage_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        input_hashes: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Records the start of a stage, writing partial status immediately to run_state.json.
        """
        curr_hashes = input_hashes if input_hashes is not None else compute_stage_input_hashes(stage_name, self.base_dir)
        composite_hash = compute_composite_hash(curr_hashes)

        if "stages" not in self._state:
            self._state["stages"] = {}

        if stage_name not in self._state["stages"]:
            spec = STAGE_DEFINITIONS.get(stage_name, {})
            self._state["stages"][stage_name] = {
                "name": stage_name,
                "display_name": spec.get("display_name", stage_name),
                "script": spec.get("script", "")
            }

        stage = self._state["stages"][stage_name]
        stage["status"] = "running"
        stage["start_time"] = datetime.datetime.now().isoformat()
        stage["end_time"] = None
        stage["duration_seconds"] = None
        stage["exit_code"] = None
        stage["error_message"] = None
        stage["input_hashes"] = curr_hashes
        stage["composite_input_hash"] = composite_hash
        stage["parameters"] = parameters or {}

        self._state["pipeline_status"] = "running"
        self.save()

    def record_stage_finish(
        self,
        stage_name: str,
        success: bool,
        exit_code: int = 0,
        error_message: Optional[str] = None
    ) -> None:
        """
        Records the completion (success or failure) of a stage.
        """
        stage = self._state.get("stages", {}).get(stage_name)
        if not stage:
            return

        end_dt = datetime.datetime.now()
        stage["end_time"] = end_dt.isoformat()
        stage["exit_code"] = exit_code
        stage["status"] = "completed" if success else "failed"
        stage["error_message"] = error_message

        # Calculate duration if start_time is available
        start_iso = stage.get("start_time")
        if start_iso:
            try:
                start_dt = datetime.datetime.fromisoformat(start_iso)
                stage["duration_seconds"] = round((end_dt - start_dt).total_seconds(), 2)
            except Exception:
                stage["duration_seconds"] = None

        # Update pipeline overall status
        if not success:
            self._state["pipeline_status"] = "failed"
        else:
            # If all defined stages are completed
            all_completed = all(
                s.get("status") in ("completed", "skipped")
                for s in self._state["stages"].values()
            )
            if all_completed:
                self._state["pipeline_status"] = "completed"

        self.save()

    def record_stage_skipped(self, stage_name: str, reason: str = "Stage skipped") -> None:
        """Records a stage as skipped."""
        stage = self._state.get("stages", {}).get(stage_name)
        if stage:
            stage["status"] = "skipped"
            stage["error_message"] = reason
            self.save()

    def reset(self) -> None:
        """Resets the state file to clean defaults."""
        self._state = self._default_state()
        self.save()
        logger.info(f"Pipeline state reset in {self.state_file}")

    def format_status_table(self) -> str:
        """
        Generates a clean pipeline status table showing stage name, status, last run time, and duration.
        """
        lines = []
        lines.append("=" * 78)
        lines.append(f" CENTROID Pipeline State ({self.state_file.name})")
        lines.append("=" * 78)
        header = f" {'Stage':<18} | {'Status':<11} | {'Last Run Time':<22} | {'Duration':<9} | {'Exit'}"
        lines.append(header)
        lines.append("-" * 78)

        for name, spec in STAGE_DEFINITIONS.items():
            stage = self.get_stage_state(name) or {}
            status = stage.get("status", "pending")
            start_iso = stage.get("start_time") or stage.get("end_time")

            if start_iso:
                try:
                    dt = datetime.datetime.fromisoformat(start_iso)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    time_str = str(start_iso)[:19]
            else:
                time_str = "Never"

            duration = stage.get("duration_seconds")
            dur_str = f"{duration:.2f}s" if duration is not None else "-"
            exit_code = stage.get("exit_code")
            exit_str = str(exit_code) if exit_code is not None else "-"

            # Visual status styling indicator
            status_display = status
            if status == "completed":
                status_display = "completed"
            elif status == "failed":
                status_display = "FAILED"
            elif status == "running":
                status_display = "RUNNING"
            elif status == "skipped":
                status_display = "skipped"
            else:
                status_display = "pending"

            row = f" {name:<18} | {status_display:<11} | {time_str:<22} | {dur_str:<9} | {exit_str}"
            lines.append(row)

        lines.append("=" * 78)
        return "\n".join(lines)

    def print_status_table(self) -> None:
        """Prints the formatted status table to standard output."""
        print("\n" + self.format_status_table() + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="CENTROID Pipeline Stage State & Checkpoint Manager."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the current pipeline stage status table."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset run_state.json to default state."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON state."
    )
    parser.add_argument(
        "--check",
        type=str,
        default=None,
        help="Check if a specific stage is current (returns 0 if current, 1 if rerun required)."
    )

    args = parser.parse_args()
    mgr = StateManager()

    if args.reset:
        mgr.reset()
        print("Pipeline state successfully reset.")
        return

    if args.json:
        print(json.dumps(mgr._state, indent=2))
        return

    if args.check:
        canonical = resolve_stage_name(args.check)
        if not canonical:
            print(f"Unknown stage: {args.check}")
            sys.exit(2)
        current, reason = mgr.is_stage_current(canonical)
        print(f"Stage '{canonical}': current={current} ({reason})")
        sys.exit(0 if current else 1)

    # Default action: print status table
    mgr.print_status_table()


if __name__ == "__main__":
    main()
