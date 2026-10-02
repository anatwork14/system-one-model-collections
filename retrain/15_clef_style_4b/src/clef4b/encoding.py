"""Reuse official state/schema serialization and span tracking."""

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]
UPSTREAM = WORKSPACE / "released/05_clef_flash_9b/weights/architecture"
sys.path.insert(0, str(UPSTREAM))

from joint_schema_model import EncodedQuestion, EncodedRecord, encode_record  # noqa: E402

__all__ = ["EncodedQuestion", "EncodedRecord", "encode_record"]
