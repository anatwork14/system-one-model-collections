"""Re-export Cloudflare's released routing-head implementation."""

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]
UPSTREAM = WORKSPACE / "released/05_clef_flash_9b/weights/architecture"
sys.path.insert(0, str(UPSTREAM))

from joint_schema_model import EvidenceRoutingLayer, JointSchemaHead  # noqa: E402

__all__ = ["EvidenceRoutingLayer", "JointSchemaHead"]
