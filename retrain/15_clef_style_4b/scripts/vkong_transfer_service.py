#!/usr/bin/env python3
"""Train M15 without VKong storage and serve its checkpoint for local download."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path("/data/checkpoints")
OUTPUT = ROOT / "clef4b-lora-clean"
ARCHIVE = ROOT / "m15-clef4b-lora.tar"
LOG = ROOT / "training.log"
PORT = 8000
lock = threading.Lock()
state: dict[str, object] = {"status": "starting"}


def set_state(**values: object) -> None:
    with lock:
        state.update(values)


def stream_process_output(
    process: subprocess.Popen,
    log,
    output=None,
) -> int:
    """Copy child output byte-for-byte to the durable log and VKong stdout."""
    if process.stdout is None:
        raise ValueError("process must be started with stdout=PIPE")
    output = output if output is not None else sys.stdout.buffer
    while True:
        chunk = os.read(process.stdout.fileno(), 4096)
        if not chunk:
            break
        log.write(chunk)
        log.flush()
        output.write(chunk)
        output.flush()
    process.stdout.close()
    return process.wait()


def train_and_package() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    set_state(status="training")
    env = os.environ.copy()
    env.update({"CLEF_STAGE": "lora", "CLEF_EPOCHS": "1", "CLEF_RANK": "256",
                "PYTHONUNBUFFERED": "1"})
    with LOG.open("wb") as log:
        process = subprocess.Popen(
            ["bash", "scripts/vkong_train.sh"], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0, env=env
        )
        exit_code = stream_process_output(process, log)
    if exit_code != 0:
        set_state(status="failed", exit_code=exit_code)
        return

    set_state(status="packaging", exit_code=0)
    with tarfile.open(ARCHIVE, "w") as archive:
        archive.add(OUTPUT, arcname=OUTPUT.name)

    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    set_state(status="ready", exit_code=0, archive=ARCHIVE.name,
              bytes=ARCHIVE.stat().st_size, sha256=digest.hexdigest())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        print("transfer-http: " + fmt % args, flush=True)

    def send_bytes(self, content_type: str, data: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/status":
            with lock:
                payload = json.dumps(state, sort_keys=True).encode()
            self.send_bytes("application/json", payload)
            return
        if self.path == "/logs":
            tail = LOG.read_bytes()[-65536:] if LOG.exists() else b""
            self.send_bytes("text/plain; charset=utf-8", tail)
            return
        if self.path == "/artifact":
            with lock:
                ready = state.get("status") == "ready"
            if not ready or not ARCHIVE.is_file():
                self.send_bytes("application/json", b'{"error":"artifact not ready"}', 409)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-tar")
            self.send_header("Content-Length", str(ARCHIVE.stat().st_size))
            self.send_header("Content-Disposition", f'attachment; filename="{ARCHIVE.name}"')
            self.end_headers()
            with ARCHIVE.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    self.wfile.write(block)
            return
        self.send_bytes("text/plain; charset=utf-8", b"Use /status, /logs, or /artifact\n", 404)


if __name__ == "__main__":
    threading.Thread(target=train_and_package, daemon=True).start()
    print(f"M15 transfer service listening on 0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
