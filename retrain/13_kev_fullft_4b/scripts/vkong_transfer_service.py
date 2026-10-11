#!/usr/bin/env python3
"""Run M13 and expose its completed checkpoint over the attached VKong tunnel."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path("/data/checkpoints")
OUTPUT = ROOT / "kev-fullft-clean"
ARCHIVE = ROOT / "kev-fullft-clean.tar"
LOG = ROOT / "training.log"
PORT = 8000
lock = threading.Lock()
state: dict[str, object] = {"status": "starting"}


def set_state(**values: object) -> None:
    with lock:
        state.update(values)


def train_and_package() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    set_state(status="training")
    with LOG.open("wb") as log:
        process = subprocess.Popen(
            ["bash", "scripts/vkong_train.sh"], stdout=log, stderr=subprocess.STDOUT
        )
        exit_code = process.wait()
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
    set_state(
        status="ready",
        exit_code=0,
        archive=str(ARCHIVE.name),
        bytes=ARCHIVE.stat().st_size,
        sha256=digest.hexdigest(),
    )


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
                data = json.dumps(state, sort_keys=True).encode()
            self.send_bytes("application/json", data)
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
    worker = threading.Thread(target=train_and_package, daemon=True)
    worker.start()
    print(f"M13 transfer service listening on 0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
