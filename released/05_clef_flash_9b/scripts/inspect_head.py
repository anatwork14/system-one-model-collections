#!/usr/bin/env python3
"""Report released Clef-Flash head tensor shapes and dtypes."""

from pathlib import Path

from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "architecture" / "joint_head.safetensors"
OUTPUT = ROOT / "audit" / "head_tensor_shapes.txt"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    total = 0
    with safe_open(WEIGHTS, framework="pt", device="cpu") as tensors:
        for name in sorted(tensors.keys()):
            view = tensors.get_slice(name)
            shape = tuple(view.get_shape())
            dtype = view.get_dtype()
            count = 1
            for dim in shape:
                count *= dim
            total += count
            lines.append(f"{name}\t{shape}\t{dtype}\t{count}")
    lines.append(f"TOTAL\tparameters={total}")
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines) - 1} tensors, {total:,} parameters to {OUTPUT}")


if __name__ == "__main__":
    main()
