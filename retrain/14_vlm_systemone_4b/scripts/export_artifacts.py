#!/usr/bin/env python3
"""Emit compact task artifacts as checksummed stdout chunks for local capture."""
import argparse
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=24000)
    args = parser.parse_args()
    names = ["best.pt", "last.pt", "run_config.json", "progress.jsonl",
             "eval/dev_predictions.jsonl", "eval/dev_predictions.jsonl.summary.json"]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in names:
            path = args.run_dir / relative
            if path.is_file():
                archive.add(path, arcname=relative, recursive=False)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    chunks = [encoded[i:i + args.chunk_bytes] for i in range(0, len(encoded), args.chunk_bytes)]
    print(json.dumps({"event": "artifact_export_start", "sha256": digest,
                      "archive_bytes": len(payload), "chunks": len(chunks)}, separators=(",", ":")), flush=True)
    for index, chunk in enumerate(chunks, 1):
        print(f"M14_EXPORT {index}/{len(chunks)} {digest} {chunk}", flush=True)
    print(json.dumps({"event": "artifact_export_complete", "sha256": digest,
                      "archive_bytes": len(payload), "chunks": len(chunks)}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
