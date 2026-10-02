"""Expose the official SystemOne-compatible response behavior."""

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]
UPSTREAM = WORKSPACE / "released/05_clef_flash_9b/weights/architecture"
sys.path.insert(0, str(UPSTREAM))

from joint_schema_model import systemone, systemone_answer  # noqa: E402

__all__ = ["systemone", "systemone_answer"]
