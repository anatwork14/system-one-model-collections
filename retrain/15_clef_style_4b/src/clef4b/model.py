"""Model classes and the fresh 4B head initialization used by the reproduction."""

import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]
UPSTREAM = WORKSPACE / "released/05_clef_flash_9b/weights/architecture"
sys.path.insert(0, str(UPSTREAM))

from joint_schema_model import ClefModel, JointSchemaHead  # noqa: E402


def fresh_head(qwen_hidden_size: int) -> JointSchemaHead:
    """Construct a newly initialized official head for a backbone width."""
    config = json.loads((UPSTREAM / "joint_head_config.json").read_text())
    config["hidden_size"] = qwen_hidden_size
    return JointSchemaHead(**config)


__all__ = ["ClefModel", "JointSchemaHead", "fresh_head"]
