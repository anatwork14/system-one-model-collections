#!/usr/bin/env python3
"""Render a compact registry JSON snapshot from the canonical YAML registry."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
registry = yaml.safe_load((ROOT / "registry.yaml").read_text())
(ROOT / "manifests" / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
print(f"Wrote {(ROOT / 'manifests' / 'registry.json').relative_to(ROOT)}")
