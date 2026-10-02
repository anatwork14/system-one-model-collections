#!/usr/bin/env python3
"""Verify the full local snapshot against its sharded safetensors index."""

import json
from pathlib import Path

from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "weights" / "model"


def main() -> None:
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    expected = set(index["weight_map"].values())
    missing = sorted(name for name in expected if not (MODEL / name).is_file())
    if missing:
        raise SystemExit(f"missing model shards: {missing}")
    tensor_bytes = 0
    expected_bytes = index["metadata"]["total_size"]
    seen = set()
    for name in sorted(expected):
        with safe_open(MODEL / name, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                seen.add(key)
                tensor = tensors.get_slice(key)
                dtype = tensor.get_dtype()
                if dtype.startswith("F8_") or dtype.startswith("U8") or dtype.startswith("I8"):
                    itemsize = 1
                elif dtype.startswith("F16") or dtype.startswith("BF16") or dtype.startswith("I16") or dtype.startswith("U16"):
                    itemsize = 2
                elif dtype.startswith("F32") or dtype.startswith("I32") or dtype.startswith("U32"):
                    itemsize = 4
                elif dtype.startswith("F64") or dtype.startswith("I64") or dtype.startswith("U64"):
                    itemsize = 8
                else:
                    raise SystemExit(f"unsupported tensor dtype for size verification: {dtype}")
                count = 1
                for dimension in tensor.get_shape():
                    count *= dimension
                tensor_bytes += count * itemsize
    indexed = set(index["weight_map"])
    if seen != indexed:
        raise SystemExit(f"tensor index mismatch: {len(indexed - seen)} missing, {len(seen - indexed)} extra")
    if tensor_bytes != expected_bytes:
        raise SystemExit(f"tensor byte size mismatch: expected {expected_bytes}, got {tensor_bytes}")
    physical_bytes = sum((MODEL / name).stat().st_size for name in expected)
    print(json.dumps({"repo": "Cloudflare/clef-flash", "weight_shards": len(expected), "shard_bytes": physical_bytes, "tensor_bytes": tensor_bytes, "tensors": len(seen), "index_verified": True}, indent=2))


if __name__ == "__main__":
    main()
