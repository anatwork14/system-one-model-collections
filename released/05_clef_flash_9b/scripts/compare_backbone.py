#!/usr/bin/env python3
"""Compare two local sharded safetensors checkpoints without downloading either."""

import argparse
import csv
import json
import math
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open


def weight_map(path: Path) -> dict[str, str]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    files = sorted(path.glob("*.safetensors"))
    result = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as tensors:
            result.update({key: file.name for key in tensors.keys()})
    return result


def load_tensor(path: Path, filename: str, name: str) -> torch.Tensor:
    with safe_open(path / filename, framework="pt", device="cpu") as tensors:
        return tensors.get_tensor(name).float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Local Qwen3.5-9B directory")
    parser.add_argument("--clef", type=Path, required=True, help="Local Clef-Flash directory")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "audit/parameter_diff.csv")
    args = parser.parse_args()
    base_map, clef_map = weight_map(args.base), weight_map(args.clef)
    rows = []
    with ExitStack() as stack:
        base_files = {name: stack.enter_context(safe_open(args.base / name, framework="pt", device="cpu")) for name in set(base_map.values())}
        clef_files = {name: stack.enter_context(safe_open(args.clef / name, framework="pt", device="cpu")) for name in set(clef_map.values())}
        for name in sorted(set(base_map) & set(clef_map)):
            before = base_files[base_map[name]].get_tensor(name).float()
            after = clef_files[clef_map[name]].get_tensor(name).float()
            if before.shape != after.shape:
                rows.append([name, str(tuple(before.shape)), "", "", "", "shape_mismatch", "", ""])
                continue
            norm = float(torch.linalg.vector_norm(before))
            delta = float(torch.linalg.vector_norm(after - before))
            relative = delta / (norm + 1e-12)
            layer = next((part for part in name.split(".") if part.isdigit()), "")
            module_type = name.rsplit(".", 1)[-1]
            rows.append([name, str(tuple(before.shape)), norm, delta, relative, "yes" if not math.isclose(relative, 0.0, abs_tol=1e-8) else "no", layer, module_type])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tensor_name", "shape", "base_norm", "delta_norm", "relative_delta", "changed", "layer", "module_type"])
        writer.writerows(rows)
    print(f"Compared {len(rows):,} matching tensors; wrote {args.output}")
    print("This script reads only the supplied local directories; it does not fetch Qwen weights.")


if __name__ == "__main__":
    main()
