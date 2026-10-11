#!/usr/bin/env python3
"""Regression tests for VKong setup helpers."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("vkong_setup.py")


def load_setup_module():
    torch_stub = types.ModuleType("torch")
    pyarrow_stub = types.ModuleType("pyarrow")
    pyarrow_stub.__path__ = []
    parquet_stub = types.ModuleType("pyarrow.parquet")
    setattr(pyarrow_stub, "parquet", parquet_stub)
    hub_stub = types.ModuleType("huggingface_hub")
    setattr(hub_stub, "hf_hub_download", None)
    setattr(hub_stub, "snapshot_download", None)
    dependencies = {
        "torch": torch_stub,
        "pyarrow": pyarrow_stub,
        "pyarrow.parquet": parquet_stub,
        "huggingface_hub": hub_stub,
    }
    spec = importlib.util.spec_from_file_location("vkong_setup_under_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load setup script: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


class Column:
    def __init__(self, payload: bytes):
        self.payload = payload

    def to_pylist(self):
        return [{"bytes": self.payload}]


class Batch:
    def __init__(self, payload: bytes):
        self.payload = payload

    def column(self, _index: int):
        return Column(self.payload)


class ReconstructImagesTest(unittest.TestCase):
    def test_returns_recovered_image_count_after_hash_verification(self):
        setup = load_setup_module()
        payload = b"audited flower image bytes"
        digest = hashlib.sha256(payload).hexdigest()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            manifest_dir = data / "manifests"
            manifest_dir.mkdir(parents=True)
            (manifest_dir / "leakage_audit.json").write_text(
                json.dumps({
                    "status": "passed",
                    "image_files": {
                        "images/flowers102/flower.jpg": {"sha256": digest}
                    },
                }),
                encoding="utf-8",
            )
            class FakeParquetFile:
                def __init__(self, _path):
                    pass

                def iter_batches(self, *, batch_size, columns):
                    yield Batch(payload)

            setattr(setup, "DATA", data)
            setattr(getattr(setup, "parquet"), "ParquetFile", FakeParquetFile)
            setattr(setup, "hf_hub_download", lambda **_kwargs: str(root / "source.parquet"))

            recovered_count = setup.reconstruct_images()

            image_path = data / "images/flowers102/flower.jpg"
            self.assertEqual(recovered_count, 1)
            self.assertEqual(image_path.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
