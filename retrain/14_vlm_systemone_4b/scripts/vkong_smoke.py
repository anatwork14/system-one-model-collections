#!/usr/bin/env python3
"""Load the pinned VLM and verify one multimodal candidate-scoring pass."""
import json
from pathlib import Path

import torch
from transformers import Qwen3_5ForConditionalGeneration

from train import DATA, DEFAULT_MAX_IMAGE_PIXELS, MODEL_PATH, candidate_features, load_processor


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in the VKong runtime")
    rows = [json.loads(line) for line in (DATA / "processed/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("M14 training split is empty")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = load_processor(DEFAULT_MAX_IMAGE_PIXELS)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True,
    ).to("cuda").eval()
    features = candidate_features(model, processor, rows[0], torch.device("cuda"), dtype, 2048)
    if features.ndim != 2 or features.shape[0] != len(rows[0]["candidates"]) or not torch.isfinite(features).all():
        raise SystemExit(f"Unexpected candidate feature tensor: {tuple(features.shape)}")
    print(json.dumps({"event": "smoke_passed", "candidate_features": list(features.shape),
                      "gpu": torch.cuda.get_device_name(0),
                      "allocated_vram_gib": round(torch.cuda.memory_allocated() / 1024**3, 2)}), flush=True)


if __name__ == "__main__":
    main()
