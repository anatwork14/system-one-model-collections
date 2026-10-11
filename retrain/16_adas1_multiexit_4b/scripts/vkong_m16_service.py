#!/usr/bin/env python3
"""Run M16 P0 on ephemeral VKong storage and serve its small result archive."""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO, cast

def workspace_root_from_service_path(service_path: Path) -> Path:
    """Find the uploaded workspace by its M16 config, independent of cwd depth."""
    resolved = service_path.resolve()
    for candidate in (resolved.parent, *resolved.parents, Path.cwd().resolve()):
        if (candidate / "retrain/16_adas1_multiexit_4b/configs/probe.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        f"cannot find uploaded M16 workspace from service path {resolved}; "
        "expected retrain/16_adas1_multiexit_4b/configs/probe.yaml"
    )


WORKSPACE_ROOT = workspace_root_from_service_path(Path(__file__))
METHOD = WORKSPACE_ROOT / "retrain/16_adas1_multiexit_4b"
CONFIG = METHOD / "configs/probe.yaml"
DATA_MANIFEST = METHOD / "data/controlled/manifest.json"
MODEL_ARCH = WORKSPACE_ROOT / "released/05_clef_flash_9b/weights/architecture"
DEFAULT_RUNTIME_ROOT = Path("/tmp/adas1-p0")
RUNTIME_ROOT = Path(os.environ.get("M16_RUNTIME_ROOT", str(DEFAULT_RUNTIME_ROOT)))
LOG = RUNTIME_ROOT / "service.log"
PORT = 8000


def make_run_id(now: datetime | None = None) -> str:
    """Generate a timestamped run ID with an unpredictable collision-resistant suffix."""
    created_at = now or datetime.now(timezone.utc)
    return f"{created_at.strftime('%Y%m%d-%H%M%S-%f')}-{uuid.uuid4().hex[:12]}"


def unique_run_id(runtime_root: Path, preferred: str) -> str:
    """Avoid reusing a persisted output directory when an App restarts on the same VM."""
    candidate = preferred
    for _ in range(8):
        archive = runtime_root / f"m16-p0-{candidate}-results.tar.gz"
        if not (runtime_root / candidate).exists() and not archive.exists():
            return candidate
        candidate = make_run_id()
    raise FileExistsError(f"could not allocate a unique M16 run directory under {runtime_root}")


RUN_ID = make_run_id()
OUTPUT = RUNTIME_ROOT / RUN_ID
ARCHIVE = RUNTIME_ROOT / f"m16-p0-{RUN_ID}-results.tar.gz"
FAILURE_GRACE_SECONDS = float(os.environ.get("M16_FAILURE_GRACE_SECONDS", "45"))
lock = threading.Lock()
state: dict[str, object] = {"status": "starting", "run_id": RUN_ID}


def configured_layers(config_path: Path = CONFIG) -> tuple[int, ...]:
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"^layers:\s*\[([^]]*)\]\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing inline layers list in {config_path}")
    layers = tuple(int(layer.strip()) for layer in match.group(1).split(",") if layer.strip())
    if not layers or tuple(sorted(set(layers))) != layers:
        raise ValueError(f"probe layers must be non-empty, unique, and increasing: {layers}")
    if any(layer < 1 or layer > 32 for layer in layers):
        raise ValueError(f"probe layers must be between 1 and 32: {layers}")
    return layers


def validate_layers_for_model(layers: tuple[int, ...], actual_layer_count: int) -> None:
    if actual_layer_count < 1:
        raise ValueError(f"model decoder layer count must be positive: {actual_layer_count}")
    invalid = [layer for layer in layers if layer > actual_layer_count]
    if invalid:
        raise ValueError(
            f"configured probe layers {invalid} exceed the model's "
            f"{actual_layer_count} decoder layers"
        )


def model_helper_path(workspace_root: Path) -> Path:
    """Resolve the required helper in its preserved or flattened upload location."""
    candidates = (
        workspace_root / "released/05_clef_flash_9b/weights/architecture/joint_schema_model.py",
        workspace_root / "joint_schema_model.py",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(path for path in workspace_root.rglob("joint_schema_model.py")
                     if ".git" not in path.parts and "__pycache__" not in path.parts)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError("multiple uploaded joint_schema_model.py files found: "
                                + ", ".join(map(str, matches)))
    raise FileNotFoundError("uploaded VKong sources omit joint_schema_model.py; checked "
                            + ", ".join(map(str, candidates)))


def model_head_config_path(workspace_root: Path) -> Path:
    """Resolve the pinned joint-head config inside the VKong source upload."""
    candidates = (
        workspace_root / "retrain/16_adas1_multiexit_4b/vendor/architecture/joint_head_config.json",
        workspace_root / "joint_head_config.json",
        workspace_root / "released/05_clef_flash_9b/weights/architecture/joint_head_config.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(path for path in workspace_root.rglob("joint_head_config.json")
                     if ".git" not in path.parts and "__pycache__" not in path.parts)
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError("uploaded VKong sources omit joint_head_config.json; checked "
                            + ", ".join(map(str, candidates)))


def set_state(**values: object) -> None:
    with lock:
        state.update(values)


def _process_group_alive(process: subprocess.Popen) -> bool:
    if process.poll() is None:
        return True
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Terminate the supervised process group, then force-kill if it ignores SIGTERM."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return
    deadline = time.monotonic() + 3.0
    while _process_group_alive(process) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _process_group_alive(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.wait()


def stream_process_output(process: subprocess.Popen, log, output=None,
                          timeout_seconds: float | None = None) -> int:
    """Stream child bytes to both sinks, terminating its process group at deadline."""
    if process.stdout is None:
        raise ValueError("process must be started with stdout=PIPE")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    pipe = cast(BinaryIO, process.stdout)
    output = output if output is not None else sys.stdout.buffer
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    timed_out = False
    with selectors.DefaultSelector() as selector:
        selector.register(pipe, selectors.EVENT_READ)
        while selector.get_map():
            if deadline is not None and time.monotonic() >= deadline:
                if _process_group_alive(process):
                    timed_out = True
                    _terminate_process_tree(process)
                deadline = None
            timeout = 0.1 if deadline is None else min(0.1, max(0.0, deadline - time.monotonic()))
            for _key, _events in selector.select(timeout):
                chunk = os.read(pipe.fileno(), 4096)
                if not chunk:
                    selector.unregister(pipe)
                    pipe.close()
                    continue
                log.write(chunk)
                log.flush()
                output.write(chunk)
                output.flush()
    process.wait()
    return 124 if timed_out else int(process.returncode)


def artifact_sources(output: Path, data_manifest: Path, layers: tuple[int, ...]) -> list[tuple[Path, Path]]:
    """Return (source, archive path) pairs, excluding reproducible feature caches."""
    sources = [
        (output / "probe_report.json", Path("probe_report.json")),
        (output / "exit_curve.csv", Path("exit_curve.csv")),
        (output / "exit_curve.svg", Path("exit_curve.svg")),
    ]
    for layer in layers:
        layer_dir = output / f"layer_{layer}"
        sources.extend((layer_dir / name, Path(f"layer_{layer}") / name)
                       for name in ("probe.pt", "history.json"))
    sources.append((data_manifest, Path("input/data/controlled/manifest.json")))
    missing = [str(path) for path, _ in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing expected M16 artifacts: " + ", ".join(missing))
    return sources


def training_artifact_sources(output: Path, data_manifest: Path, stage: str) -> list[tuple[Path, Path]]:
    """Return every durable training artifact plus the exact controlled-data manifest."""
    required = {
        "teacher": (Path("teacher_head.pt"),),
        "multiexit": (Path("exit_heads.pt"), Path("teacher_head.pt")),
        "full": (Path("teacher/teacher_head.pt"), Path("multiexit/exit_heads.pt")),
    }
    if stage not in required:
        raise ValueError(f"unsupported M16 training stage: {stage}")
    missing_weights = [str(output / relative) for relative in required[stage]
                       if not (output / relative).is_file()]
    if missing_weights:
        raise FileNotFoundError("missing expected M16 training weights: " + ", ".join(missing_weights))
    sources = [
        (path, Path("training") / stage / path.relative_to(output))
        for path in sorted(output.rglob("*")) if path.is_file() and not path.is_symlink()
    ]
    sources.append((data_manifest, Path("input/data/controlled/manifest.json")))
    return sources


def partial_training_artifact_sources(
    output: Path, data_manifest: Path, log_path: Path, stage: str,
) -> list[tuple[Path, Path]]:
    """Collect completed checkpoints, partial reports, logs, and manifest after failure."""
    sources = [
        (path, Path("training") / stage / path.relative_to(output))
        for path in sorted(output.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name != "partial_artifact_manifest.json"
    ]
    if data_manifest.is_file():
        sources.append((data_manifest, Path("input/data/controlled/manifest.json")))
    if log_path.is_file():
        sources.append((log_path, Path("logs/service.log")))
    return sources


def package_partial_training_artifacts(
    output: Path, data_manifest: Path, log_path: Path, archive_path: Path,
    run_id: str, stage: str,
) -> dict[str, object]:
    """Write a checksummed archive of recoverable files after a failed training stage."""
    sources = partial_training_artifact_sources(output, data_manifest, log_path, stage)
    if not sources:
        return {"partial_artifacts": False}
    entries = [{
        "path": (Path("M16-TRAIN") / run_id / archive_name).as_posix(),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    } for source, archive_name in sources]
    manifest_path = output / "partial_artifact_manifest.json"
    manifest_path.write_text(json.dumps({
        "run_id": run_id, "stage": stage, "status": "failed", "artifacts": entries,
    }, indent=2) + "\n", encoding="utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        for source, archive_name in sources:
            archive.add(source, arcname=(Path("M16-TRAIN") / run_id / archive_name).as_posix())
        archive.add(manifest_path, arcname=(Path("M16-TRAIN") / run_id / "partial_artifact_manifest.json").as_posix())
    return {
        "partial_artifacts": True,
        "archive": archive_path.name,
        "archive_path": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "artifact_count": len(entries),
        "artifact_manifest": "partial_artifact_manifest.json",
        "partial_weights": any(Path(name).suffix in {".pt", ".safetensors"} for _, name in sources),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model_assets(workspace_root: Path) -> tuple[Path, Path]:
    """Resolve and verify both serialized head code and config against ARTIFACTS.lock."""
    helper = model_helper_path(workspace_root)
    head_config = model_head_config_path(workspace_root)
    lock_candidates = (
        workspace_root / "ARTIFACTS.lock",
        workspace_root / "retrain/16_adas1_multiexit_4b/ARTIFACTS.lock",
    )
    lock_path = next((path for path in lock_candidates if path.is_file()), None)
    if lock_path is None:
        raise FileNotFoundError("uploaded VKong sources omit ARTIFACTS.lock")
    pinned = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator:
            pinned[key.strip()] = value.strip()
    for key, path, label in (
        ("serialization_sha256", helper, "joint_schema_model.py"),
        ("head_config_sha256", head_config, "head config"),
    ):
        expected = pinned.get(key)
        if expected is None or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError(f"missing or invalid {key} pin in {lock_path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{label} checksum mismatch: expected {expected}, got {actual}")
    return helper, head_config


def validate_training_inputs(workspace_root: Path) -> dict:
    """Verify canonical train/dev files against both their manifest and artifact lock."""
    method = workspace_root / "retrain/16_adas1_multiexit_4b"
    data_dir = method / "data/controlled"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    lock_path = method / "ARTIFACTS.lock"
    pins = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator:
            pins[key.strip()] = value.strip()
    if manifest.get("dataset") != "nimble-m16-p0-controlled-v1":
        raise ValueError("unexpected controlled M16 dataset")
    if manifest.get("outer_324_eval_read") is not False:
        raise ValueError("manifest must assert that outer test was not read")
    if manifest.get("family_disjoint") is not True or manifest.get("source_id_disjoint") is not True:
        raise ValueError("manifest must assert train/dev family and source-id disjointness")
    source_hashes = manifest.get("source_sha256", {})
    controlled_hashes = manifest.get("controlled_sha256", {})
    for split in ("train", "dev"):
        expected_source = pins.get(f"{split}_source_sha256")
        if source_hashes.get(split) != expected_source or not expected_source:
            raise ValueError(f"{split} source hash does not match ARTIFACTS.lock")
        path = data_dir / f"{split}.jsonl"
        actual = sha256_file(path)
        if actual != controlled_hashes.get(split):
            raise ValueError(f"controlled {split} checksum mismatch with manifest")
        if actual != pins.get(f"controlled_{split}_sha256"):
            raise ValueError(f"controlled {split} checksum mismatch with ARTIFACTS.lock")
        if manifest.get("counts", {}).get(split) != int(pins.get(f"controlled_{split}_count", -1)):
            raise ValueError(f"controlled {split} count mismatch with ARTIFACTS.lock")
    return manifest


def build_probe_command(workspace_root: Path, output: Path, layers: tuple[int, ...]) -> list[str]:
    runner = workspace_root / "retrain/16_adas1_multiexit_4b/scripts/vkong_run_probe.sh"
    return ["bash", str(runner), "--layers", ",".join(map(str, layers)), "--output", str(output)]


def build_training_command(
    workspace_root: Path,
    output: Path,
    stage: str,
    exit_layers: tuple[int, ...] = (12, 16, 20),
    *,
    max_epochs: int = 3,
    max_runtime_seconds: int = 14400,
    teacher_checkpoint: Path | None = None,
) -> list[str]:
    if stage not in {"teacher", "multiexit", "full"}:
        raise ValueError(f"unsupported M16 training stage: {stage}")
    if not exit_layers or tuple(sorted(set(exit_layers))) != exit_layers:
        raise ValueError("M16 exit layers must be nonempty, unique, and increasing")
    if max_epochs < 1 or max_runtime_seconds < 1:
        raise ValueError("M16 training epoch and runtime limits must be positive")
    if stage == "multiexit" and teacher_checkpoint is None:
        raise ValueError("multiexit stage requires a teacher checkpoint")
    trainer = workspace_root / "retrain/16_adas1_multiexit_4b/scripts/train_m16.py"
    command = [sys.executable, "-u", str(trainer), "--stage", stage,
               "--output", str(output), "--exit-layers", ",".join(map(str, exit_layers)),
               "--max-epochs", str(max_epochs), "--max-runtime-seconds", str(max_runtime_seconds)]
    if teacher_checkpoint is not None:
        command.extend(("--teacher-checkpoint", str(teacher_checkpoint)))
    return command


def _fail_and_shutdown(server: ThreadingHTTPServer | None, exit_code: int) -> None:
    if server is not None:
        time.sleep(max(0.0, FAILURE_GRACE_SECONDS))
        server.shutdown()


def train_and_package(
    probe_command: list[str] | None = None,
    workspace_root: Path = WORKSPACE_ROOT,
    runtime_root: Path = RUNTIME_ROOT,
    server: ThreadingHTTPServer | None = None,
    training_stage: str | None = None,
) -> int:
    """Run the configured M16 stage, publish artifacts, and propagate failures."""
    stage = training_stage or os.environ.get("M16_SERVICE_STAGE", "probe")
    run_id = unique_run_id(runtime_root, RUN_ID)
    output = runtime_root / run_id
    archive_path = runtime_root / f"m16-p0-{run_id}-results.tar.gz"
    log_path = runtime_root / "service.log"
    data_manifest = workspace_root / "retrain/16_adas1_multiexit_4b/data/controlled/manifest.json"
    try:
        runtime_root.mkdir(parents=True, exist_ok=True)
        if stage not in {"probe", "teacher", "multiexit", "full"}:
            raise ValueError(f"unsupported M16 service stage: {stage}")
        if stage == "probe":
            exit_layers = ()
            max_epochs = 0
            max_runtime_seconds = 0
            teacher_checkpoint = None
            archive_prefix = Path("M16-P0")
            archive_path = runtime_root / f"m16-p0-{run_id}-results.tar.gz"
        else:
            try:
                exit_layers = tuple(int(value.strip()) for value in os.environ.get(
                    "M16_EXIT_LAYERS", "12,16,20").split(",") if value.strip())
                max_epochs = int(os.environ.get("M16_MAX_EPOCHS", "3"))
                max_runtime_seconds = int(os.environ.get("M16_MAX_RUNTIME_SECONDS", "14400"))
            except ValueError as exc:
                raise ValueError("M16 exit layers and limits must be valid integers") from exc
            if not exit_layers or tuple(sorted(set(exit_layers))) != exit_layers or any(
                layer < 1 or layer >= 32 for layer in exit_layers
            ):
                raise ValueError("M16 student exits must be unique increasing layers from 1 to 31")
            if max_epochs < 1 or max_runtime_seconds < 1:
                raise ValueError("M16 epoch and runtime limits must be positive")
            teacher_checkpoint = os.environ.get("M16_TEACHER_CHECKPOINT")
            if stage == "multiexit" and not teacher_checkpoint:
                raise ValueError("M16_TEACHER_CHECKPOINT is required for multiexit service stage")
            archive_prefix = Path("M16-TRAIN")
            archive_path = runtime_root / f"m16-{stage}-{run_id}-results.tar.gz"
        layers = configured_layers(workspace_root / "retrain/16_adas1_multiexit_4b/configs/probe.yaml")
        helper = model_helper_path(workspace_root)
        lock_path = workspace_root / "retrain/16_adas1_multiexit_4b/ARTIFACTS.lock"
        lock_text = lock_path.read_text(encoding="utf-8")
        checksum = re.search(r"^serialization_sha256:\s*([0-9a-f]{64})\s*$", lock_text, re.MULTILINE)
        if checksum is None:
            raise ValueError(f"missing serialization_sha256 pin in {lock_path}")
        helper_hash = sha256_file(helper)
        if helper_hash != checksum.group(1):
            raise ValueError(f"joint_schema_model.py checksum mismatch: expected {checksum.group(1)}, got {helper_hash}")
        helper, head_config = validate_model_assets(workspace_root)
        if stage != "probe":
            validate_training_inputs(workspace_root)
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "M16_OUTPUT": str(output),
                    "M16_MODEL_HELPER": str(helper), "M16_HEAD_CONFIG": str(head_config)})
        active_layers = layers if stage == "probe" else exit_layers
        set_state(status="training", stage=stage, run_id=run_id, output=str(output),
                  log_path=str(log_path), layers=list(active_layers))
        if stage == "probe":
            command = probe_command or build_probe_command(workspace_root, output, layers)
        else:
            checkpoint_path = Path(teacher_checkpoint) if stage == "multiexit" and teacher_checkpoint else None
            command = probe_command or build_training_command(
                workspace_root, output, stage, exit_layers, max_epochs=max_epochs,
                max_runtime_seconds=max_runtime_seconds, teacher_checkpoint=checkpoint_path,
            )
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command, cwd=workspace_root, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=0, env=env, start_new_session=True,
            )
            exit_code = stream_process_output(
                process, log, timeout_seconds=max_runtime_seconds if stage != "probe" else None,
            )
        if exit_code != 0:
            error = (f"{stage} exceeded max runtime {max_runtime_seconds}s; process tree terminated"
                     if exit_code == 124 and stage != "probe"
                     else f"{stage} command exited with status {exit_code}")
            if stage != "probe":
                partial_archive = runtime_root / f"m16-{stage}-{run_id}-partial-results.tar.gz"
                partial_state = package_partial_training_artifacts(
                    output, data_manifest, log_path, partial_archive, run_id, stage,
                )
                set_state(status="failed", stage=stage, exit_code=exit_code,
                          error=error, **partial_state)
            else:
                set_state(status="failed", stage=stage, exit_code=exit_code, error=error)
            _fail_and_shutdown(server, exit_code)
            return exit_code

        set_state(status="packaging", stage=stage, exit_code=0)
        sources = (artifact_sources(output, data_manifest, layers) if stage == "probe"
                   else training_artifact_sources(output, data_manifest, stage))
        entries = [{"path": (archive_prefix / run_id / archive_name).as_posix(),
                    "bytes": source.stat().st_size, "sha256": sha256_file(source)}
                   for source, archive_name in sources]
        artifact_manifest = output / "artifact_manifest.json"
        artifact_manifest.write_text(json.dumps({"run_id": run_id, "artifacts": entries}, indent=2) + "\n",
                                     encoding="utf-8")
        with tarfile.open(archive_path, "w:gz") as archive:
            for source, archive_name in sources:
                archive.add(source, arcname=(archive_prefix / run_id / archive_name).as_posix())
            archive.add(artifact_manifest, arcname=(archive_prefix / run_id / "artifact_manifest.json").as_posix())

        archive_size = archive_path.stat().st_size
        archive_hash = sha256_file(archive_path)
        set_state(status="ready", stage=stage, exit_code=0, archive=archive_path.name,
                  archive_path=str(archive_path), bytes=archive_size, sha256=archive_hash,
                  artifact_count=len(entries),
                  artifact_manifest="artifact_manifest.json")
        print(f"M16 result archive ready: {archive_path.name} ({archive_size} bytes; sha256={archive_hash})",
              flush=True)
        return 0
    except Exception as exc:
        set_state(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
        print(f"M16 service failure: {type(exc).__name__}: {exc}", flush=True)
        _fail_and_shutdown(server, 1)
        return 1


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        print("transfer-http: " + format % args, flush=True)

    def send_bytes(self, content_type: str, data: bytes, status_code: int = 200) -> None:
        self.send_response(status_code)
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
            with lock:
                log_path = Path(str(state.get("log_path", LOG)))
            tail = log_path.read_bytes()[-65536:] if log_path.exists() else b""
            self.send_bytes("text/plain; charset=utf-8", tail)
            return
        if self.path == "/artifact":
            with lock:
                ready = state.get("status") == "ready"
                recoverable_partial = (
                    state.get("status") == "failed" and state.get("partial_artifacts") is True
                )
                archive_path = Path(str(state.get("archive_path", ARCHIVE)))
            if not (ready or recoverable_partial) or not archive_path.is_file():
                self.send_bytes("application/json", b'{"error":"artifact not ready"}', 409)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/gzip")
            self.send_header("Content-Length", str(archive_path.stat().st_size))
            self.send_header("Content-Disposition", f'attachment; filename="{archive_path.name}"')
            self.end_headers()
            with archive_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    self.wfile.write(block)
            return
        self.send_bytes("text/plain; charset=utf-8", b"Use /status, /logs, or /artifact\n", 404)


def serve(server: ThreadingHTTPServer | None = None, probe_command: list[str] | None = None,
          workspace_root: Path = WORKSPACE_ROOT, runtime_root: Path = RUNTIME_ROOT) -> int:
    """Serve status and artifacts; exit the service nonzero when the probe fails."""
    httpd = server or ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    worker = threading.Thread(target=train_and_package,
                              kwargs={"probe_command": probe_command, "workspace_root": workspace_root,
                                      "runtime_root": runtime_root, "server": httpd}, daemon=True)
    worker.start()
    httpd.serve_forever()
    worker.join()
    with lock:
        return int(state.get("exit_code", 1))


if __name__ == "__main__":
    print(f"M16 artifact service listening on 0.0.0.0:{PORT} run_id={RUN_ID}", flush=True)
    raise SystemExit(serve())
