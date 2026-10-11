#!/usr/bin/env python3
"""Run one bounded M17 VKong job and export its artifacts over the attach tunnel."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path("/data/checkpoints/m17_openjev_rlcd")
EXPERIMENT = Path(__file__).resolve().parents[1]
LOG = ROOT / "service.log"
ARCHIVE = ROOT / "m17-artifacts.tar.gz"
PORT = 8000
lock = threading.Lock()
state: dict[str, object] = {"status": "starting", "started_at": time.time()}


def set_state(**values):
    with lock:
        state.update(values)


def package_outputs():
    candidates = [p for p in ROOT.glob("run_*") if p.is_dir()]
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for path in candidates:
            archive.add(path, arcname=path.name)
        for path in (ROOT / "preflight.json",):
            if path.exists():
                archive.add(path, arcname=path.name)
        environment_manifest = EXPERIMENT / "manifests/environment.json"
        if environment_manifest.is_file():
            archive.add(environment_manifest, arcname="manifests/environment.json")
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"archive": ARCHIVE.name, "bytes": ARCHIVE.stat().st_size,
            "sha256": digest.hexdigest()}


def worker():
    ROOT.mkdir(parents=True, exist_ok=True)
    set_state(status="running")
    env = os.environ.copy()
    env["M17_RUN_DIR"] = str(ROOT / ("run_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())))
    env["PYTHONUNBUFFERED"] = "1"
    with LOG.open("wb") as log:
        process = subprocess.Popen(["bash", "scripts/run_m17_qwen35_4b.sh"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   env=env, bufsize=0)
        assert process.stdout is not None
        while True:
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        code = process.wait()
    set_state(status="packaging", exit_code=code)
    try:
        data = package_outputs()
        set_state(status="complete" if code == 0 else "failed", exit_code=code, **data)
    except Exception as exc:
        set_state(status="failed", exit_code=code, packaging_error=repr(exc))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("m17-transfer: " + fmt % args, flush=True)

    def send_bytes(self, content_type, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/status":
            with lock:
                payload = json.dumps(state, sort_keys=True).encode()
            return self.send_bytes("application/json", payload)
        if self.path == "/logs":
            tail = LOG.read_bytes()[-131072:] if LOG.exists() else b""
            return self.send_bytes("text/plain; charset=utf-8", tail)
        if self.path == "/artifact":
            with lock:
                ready = state.get("status") in {"complete", "failed"}
            if not ready or not ARCHIVE.is_file():
                return self.send_bytes("application/json", b'{"error":"artifact not ready"}', 409)
            self.send_response(200)
            self.send_header("Content-Type", "application/gzip")
            self.send_header("Content-Length", str(ARCHIVE.stat().st_size))
            self.send_header("Content-Disposition", f'attachment; filename="{ARCHIVE.name}"')
            self.end_headers()
            with ARCHIVE.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    self.wfile.write(block)
            return
        return self.send_bytes("text/plain", b"Use /status, /logs, or /artifact\n", 404)


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    print(f"M17 transfer service listening on 0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
