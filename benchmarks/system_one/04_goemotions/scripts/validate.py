#!/usr/bin/env python3
"""Per-dataset validate entry point."""
from pathlib import Path
import runpy
import sys
action_script = Path(__file__).resolve().parents[3] / "scripts" / "dataset_action.py"
sys.argv = [str(action_script), "--dataset", "goemotions", "--action", "validate", *sys.argv[1:]]
runpy.run_path(str(action_script), run_name="__main__")
