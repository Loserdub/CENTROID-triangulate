import os
from pathlib import Path
import subprocess
import sys

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# If .venv exists and we are not currently running in it, re-execute with .venv python
venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
if not venv_python.exists():
    venv_python = BASE_DIR / ".venv" / "bin" / "python"

if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
    cmd = [str(venv_python), str(BASE_DIR / "pipeline" / "run.py")] + sys.argv[1:]
    proc = subprocess.run(cmd, cwd=str(BASE_DIR))
    sys.exit(proc.returncode)

# Ensure pipeline package/module is in Python path
sys.path.insert(0, str(BASE_DIR))

from pipeline.run import main

if __name__ == "__main__":
    main()
