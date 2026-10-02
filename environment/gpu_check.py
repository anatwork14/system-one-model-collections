#!/usr/bin/env python3
"""Report runtime/library availability. Diagnostic only; performs no model load."""
import importlib.util
import json
import platform
import shutil
import subprocess
import sys

def command_version(command):
    if not shutil.which(command):
        return None
    try:
        p = subprocess.run([command, "--version"], text=True, capture_output=True, timeout=5)
        return (p.stdout or p.stderr).strip().splitlines()[0]
    except Exception as exc:
        return "error: " + str(exc)

report = {
    "python": sys.version,
    "platform": platform.platform(),
    "commands": {name: command_version(name) for name in ("git", "git-lfs", "gh", "nvidia-smi", "hf")},
    "modules": {name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "huggingface_hub", "peft", "safetensors")},
}
if shutil.which("nvidia-smi"):
    p = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True, capture_output=True, timeout=10)
    report["gpu_query"] = p.stdout.strip() if p.returncode == 0 else p.stderr.strip()
print(json.dumps(report, indent=2))
