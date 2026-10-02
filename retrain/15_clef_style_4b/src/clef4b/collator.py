"""Reuse official variable-question and multimodal batch collation."""

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]
UPSTREAM = WORKSPACE / "released/05_clef_flash_9b/weights/architecture"
sys.path.insert(0, str(UPSTREAM))

from joint_schema_model import collate_records  # noqa: E402

__all__ = ["collate_records"]
