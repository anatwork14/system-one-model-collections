#!/usr/bin/env python3
"""Regression checks for M16 VKong source layout, packaging, and service failures."""

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vkong_m16_service as service


class M16VkongServiceTest(unittest.TestCase):
    def test_upload_roots_preserve_runtime_source_layout_without_model_weights(self):
        repo = Path(__file__).resolve().parents[3]
        config_text = (repo / "vkong.yaml").read_text(encoding="utf-8")
        source_block = re.search(r"^srcs:\n((?:  - .*\n)+)", config_text, re.MULTILINE)
        self.assertIsNotNone(source_block)
        source_roots = [Path(line.strip()[2:]) for line in source_block.group(1).splitlines()]
        staged_helper = Path("retrain/16_adas1_multiexit_4b/vendor/architecture/joint_schema_model.py")
        self.assertIn(Path("retrain/16_adas1_multiexit_4b"), source_roots)
        self.assertNotIn(Path("released/05_clef_flash_9b/weights/architecture/joint_schema_model.py"), source_roots)
        self.assertTrue((repo / staged_helper).is_file())
        lock_text = (repo / "retrain/16_adas1_multiexit_4b/ARTIFACTS.lock").read_text(encoding="utf-8")
        pinned_hashes = re.findall(r"^serialization_sha256:\s*([0-9a-f]{64})\s*$", lock_text, re.MULTILINE)
        self.assertEqual(len(pinned_hashes), 1)
        self.assertEqual(hashlib.sha256((repo / staged_helper).read_bytes()).hexdigest(),
                         pinned_hashes[0] if pinned_hashes else None)
        self.assertNotIn(Path("released/05_clef_flash_9b/weights/architecture"), source_roots)
        self.assertIn("start: bash retrain/16_adas1_multiexit_4b/scripts/vkong_start.sh", config_text)
        for relative in (
            "retrain/16_adas1_multiexit_4b/scripts/vkong_m16_service.py",
            "retrain/16_adas1_multiexit_4b/scripts/00_probe_all_layers.py",
            "retrain/16_adas1_multiexit_4b/configs/probe.yaml",
            "retrain/16_adas1_multiexit_4b/data/controlled/manifest.json",
            staged_helper.as_posix(),
        ):
            relative_path = Path(relative)
            self.assertTrue(any(
                ((repo / root).is_file() and root == relative_path)
                or ((repo / root).is_dir() and root in relative_path.parents
                    and (repo / root / relative_path.relative_to(root)).is_file())
                for root in source_roots
            ), relative)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkout"
            method = root / "retrain/16_adas1_multiexit_4b"
            (method / "configs").mkdir(parents=True)
            (method / "configs/probe.yaml").write_text("layers: [4]\n", encoding="utf-8")
            fake_service = method / "scripts/vkong_m16_service.py"
            fake_service.parent.mkdir()
            fake_service.touch()
            self.assertEqual(service.workspace_root_from_service_path(fake_service), root)

    def test_m16_runtime_matches_m15_and_cuda_smoke_precedes_installs(self):
        repo = Path(__file__).resolve().parents[3]
        m15_config = (repo / "retrain/15_clef_style_4b/vkong.yaml").read_text(encoding="utf-8")
        m16_config = (repo / "vkong.yaml").read_text(encoding="utf-8")
        image_pattern = re.compile(r"^image:\s*(\S+)", re.MULTILINE)
        m15_image = image_pattern.search(m15_config)
        m16_image = image_pattern.search(m16_config)
        if m15_image is None or m16_image is None:
            self.fail("both M15 and M16 VKong configs must declare an image")
        self.assertEqual(m16_image.group(1), m15_image.group(1))

        setup = (repo / "retrain/16_adas1_multiexit_4b/scripts/vkong_setup.sh").read_text(encoding="utf-8")
        smoke_marker = setup.index("CUDA PREFLIGHT BEGIN")
        first_install = setup.index("python3 -m pip install")
        self.assertLess(smoke_marker, first_install)
        self.assertIn("torch.cuda.init()", setup)
        self.assertIn("torch.cuda.synchronize()", setup)
        self.assertIn("torch.empty", setup)

    def test_causal_conv1d_uses_torch_compatible_prebuilt_wheel(self):
        setup = (Path(__file__).resolve().parent / "vkong_setup.sh").read_text(encoding="utf-8")
        wheel_name = "causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
        self.assertIn("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/" + wheel_name, setup)
        self.assertIn("torch._C._GLIBCXX_USE_CXX11_ABI", setup)
        self.assertNotIn("pip install causal-conv1d==1.7.0 --no-build-isolation", setup)

    def test_fla_runtime_dependency_selects_cuda_extra(self):
        requirements = (Path(__file__).resolve().parents[1] / "requirements-runtime.txt").read_text(encoding="utf-8")
        self.assertIn("flash-linear-attention[cuda]==0.5.2", requirements)

    def test_setup_installs_c_compiler_before_triton_preflight(self):
        setup = (Path(__file__).resolve().parent / "vkong_setup.sh").read_text(encoding="utf-8")
        cuda_smoke = setup.index("CUDA PREFLIGHT BEGIN")
        compiler_install = setup.index("build-essential")
        dependency_install = setup.index("python3 -m pip install -r")
        triton_smoke = setup.index("triton.runtime.driver.active.get_current_target()")
        self.assertLess(cuda_smoke, compiler_install)
        self.assertLess(compiler_install, dependency_install)
        self.assertLess(dependency_install, triton_smoke)

    def test_setup_requires_triton_cuda_backend(self):
        setup = (Path(__file__).resolve().parent / "vkong_setup.sh").read_text(encoding="utf-8")
        self.assertIn("triton.runtime.driver.active.get_current_target()", setup)
        self.assertIn("target.backend != \"cuda\"", setup)
        self.assertIn("Triton CUDA preflight failed", setup)

    def test_p0_uses_runtime_memory_snapshot_instead_of_fixed_vram_gate(self):
        probe = (Path(__file__).resolve().parent / "00_probe_all_layers.py").read_text(encoding="utf-8")
        self.assertNotIn("MIN_GPU_MEMORY_", probe)
        self.assertIn("torch.cuda.mem_get_info(0)", probe)
        self.assertIn("memory_fit_check", probe)

    def test_cuda_compat_library_dirs_are_removed_from_runtime_path(self):
        repo = Path(__file__).resolve().parents[3]
        helper = repo / "retrain/16_adas1_multiexit_4b/scripts/vkong_cuda_compat.sh"
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/usr/lib:/usr/local/cuda/compat:/opt/cuda-compat-12-8/lib:/opt/app/lib"
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; sanitize_cuda_compat_path; printf "%s" "$LD_LIBRARY_PATH"',
             "test", str(helper)],
            env=env, check=True, capture_output=True, text=True,
        )
        self.assertEqual(result.stdout, "/usr/lib:/opt/app/lib")
        self.assertIn("Removed CUDA compatibility library path entries", result.stderr)

    def test_runtime_is_ephemeral_and_helper_upload_can_be_nested(self):
        self.assertEqual(service.DEFAULT_RUNTIME_ROOT, Path("/tmp/adas1-p0"))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nested_helper = root / "uploaded-source/architecture/joint_schema_model.py"
            nested_helper.parent.mkdir(parents=True)
            nested_helper.write_text("# uploaded helper\n", encoding="utf-8")
            self.assertEqual(service.model_helper_path(root), nested_helper)

    def test_head_configuration_is_uploaded_and_hash_pinned(self):
        repo = Path(__file__).resolve().parents[3]
        config = repo / "retrain/16_adas1_multiexit_4b/vendor/architecture/joint_head_config.json"
        self.assertTrue(config.is_file())
        lock_text = (repo / "retrain/16_adas1_multiexit_4b/ARTIFACTS.lock").read_text(encoding="utf-8")
        pinned = re.search(r"^head_config_sha256:\\s*([0-9a-f]{64})\\s*$", lock_text, re.MULTILINE)
        pinned = next((line.split(":", 1)[1].strip() for line in lock_text.splitlines()
                       if line.startswith("head_config_sha256:")), None)
        self.assertIsNotNone(pinned)
        self.assertEqual(hashlib.sha256(config.read_bytes()).hexdigest(), pinned)
        self.assertTrue(hasattr(service, "model_head_config_path"), "service does not resolve uploaded head config")
        self.assertEqual(service.model_head_config_path(repo), config)

    def test_validate_model_assets_rejects_modified_head_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            helper = root / "joint_schema_model.py"
            helper.write_text("# approved helper\\n", encoding="utf-8")
            config = root / "joint_head_config.json"
            config.write_text("{}\\n", encoding="utf-8")
            lock = root / "ARTIFACTS.lock"
            lock.write_text(
                "serialization_sha256: " + hashlib.sha256(helper.read_bytes()).hexdigest() + "\\n"
                + "head_config_sha256: " + hashlib.sha256(config.read_bytes()).hexdigest() + "\\n",
                encoding="utf-8",
            )
            lock.write_text(lock.read_text(encoding="utf-8").replace(chr(92) + "n", chr(10)),
                            encoding="utf-8")
            config.write_text("{\\\"layers\\\": 9}\\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "head config checksum mismatch"):
                service.validate_model_assets(root)

    def test_configured_layers_drive_probe_command_and_match_probe_code(self):
        layers = service.configured_layers()
        probe_source = (Path(__file__).resolve().parent / "00_probe_all_layers.py").read_text(encoding="utf-8")
        match = re.search(r"^LAYERS = \(([^)]*)\)$", probe_source, re.MULTILINE)
        self.assertIsNotNone(match)
        code_layers = tuple(int(value.strip()) for value in match.group(1).split(",") if value.strip())
        self.assertEqual(layers, code_layers)
        command = service.build_probe_command(Path("/uploaded/repo"), Path("/data/run"), layers)
        self.assertEqual(command[1], "/uploaded/repo/retrain/16_adas1_multiexit_4b/scripts/vkong_run_probe.sh")
        self.assertIn(("--layers", "4,8,12,16,20,24,28,32"),
                      list(zip(command, command[1:])))
        self.assertEqual(command[-1], "/data/run")
        service.validate_layers_for_model(layers, 32)
        with self.assertRaisesRegex(ValueError, "exceed the model's 31 decoder layers"):
            service.validate_layers_for_model(layers, 31)

    def test_validate_training_inputs_checks_pinned_canonical_splits_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "retrain/16_adas1_multiexit_4b/data/controlled"
            data.mkdir(parents=True)
            train_bytes = b'{"id":"tr","family":"train","source_id":"src-tr"}' + bytes([10])
            dev_bytes = b'{"id":"dv","family":"dev","source_id":"src-dv"}' + bytes([10])
            (data / "train.jsonl").write_bytes(train_bytes)
            (data / "dev.jsonl").write_bytes(dev_bytes)
            (data / "outer_test.jsonl").write_text("must not be read", encoding="utf-8")
            train_hash = hashlib.sha256(train_bytes).hexdigest()
            dev_hash = hashlib.sha256(dev_bytes).hexdigest()
            source_train_hash = "a" * 64
            source_dev_hash = "b" * 64
            (data / "manifest.json").write_text(json.dumps({
                "dataset": "nimble-m16-p0-controlled-v1",
                "counts": {"train": 1, "dev": 1},
                "source_sha256": {"train": source_train_hash, "dev": source_dev_hash},
                "controlled_sha256": {"train": train_hash, "dev": dev_hash},
                "family_disjoint": True, "source_id_disjoint": True,
                "outer_324_eval_read": False,
            }), encoding="utf-8")
            lock = root / "retrain/16_adas1_multiexit_4b/ARTIFACTS.lock"
            lock.write_text(
                "train_source_sha256: " + source_train_hash + chr(10)
                + "dev_source_sha256: " + source_dev_hash + chr(10)
                + "controlled_train_sha256: " + train_hash + chr(10)
                + "controlled_dev_sha256: " + dev_hash + chr(10)
                + "controlled_train_count: 1" + chr(10)
                + "controlled_dev_count: 1" + chr(10), encoding="utf-8")
            self.assertEqual(service.validate_training_inputs(root)["counts"], {"train": 1, "dev": 1})
            (data / "dev.jsonl").write_bytes(b"tampered" + bytes([10]))
            with self.assertRaisesRegex(ValueError, "controlled dev checksum mismatch"):
                service.validate_training_inputs(root)

    def test_full_stage_failure_packages_recoverable_teacher_weights(self):
        repo = Path(__file__).resolve().parents[3]
        service.state.clear()
        code = (
            "import os, pathlib, sys; "
            "root=pathlib.Path(os.environ['M16_OUTPUT']); "
            "teacher=root/'teacher'; teacher.mkdir(parents=True); "
            "(teacher/'teacher_head.pt').write_bytes(b'teacher-checkpoint'); "
            "os.write(1,b'multiexit failed\\n'); sys.exit(23)"
        )
        with tempfile.TemporaryDirectory() as temporary:
            exit_code = service.train_and_package(
                [sys.executable, "-c", code], repo, Path(temporary), training_stage="full"
            )
            self.assertEqual(exit_code, 23)
            self.assertEqual(service.state["status"], "failed")
            self.assertTrue(service.state["partial_artifacts"])
            archive_path = Path(str(service.state["archive_path"]))
            self.assertTrue(archive_path.is_file())
            with tarfile.open(archive_path, "r:gz") as archive:
                names = set(archive.getnames())
                teacher_name = next(name for name in names if name.endswith("/teacher/teacher_head.pt"))
                self.assertEqual(archive.extractfile(teacher_name).read(), b"teacher-checkpoint")
                self.assertTrue(any(name.endswith("/input/data/controlled/manifest.json") for name in names))
                partial_manifest_name = next(
                    name for name in names if name.endswith("/partial_artifact_manifest.json")
                )
                partial_manifest = json.load(archive.extractfile(partial_manifest_name))
                for entry in partial_manifest["artifacts"]:
                    payload = archive.extractfile(entry["path"]).read()
                    self.assertEqual(len(payload), entry["bytes"])
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
                self.assertFalse(any("outer_test" in name for name in names))

    def test_failed_partial_archive_is_available_from_artifact_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "partial.tar.gz"
            archive_path.write_bytes(b"partial-artifact-bytes")
            service.state.clear()
            service.state.update({
                "status": "failed", "partial_artifacts": True,
                "archive_path": str(archive_path),
            })
            server = ThreadingHTTPServer(("127.0.0.1", 0), service.Handler)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with urlopen(f"http://127.0.0.1:{server.server_address[1]}/artifact") as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.read(), b"partial-artifact-bytes")
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=2)

    def test_training_service_validates_inputs_before_launch(self):
        repo = Path(__file__).resolve().parents[3]
        service.state.clear()
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            with mock.patch.object(
                service, "validate_training_inputs",
                side_effect=ValueError("controlled split check failed"),
            ), mock.patch.object(
                service.subprocess, "Popen",
                side_effect=AssertionError("training process launched before input validation"),
            ):
                exit_code = service.train_and_package(
                    ["unused"], repo, runtime, training_stage="teacher"
                )
        self.assertEqual(exit_code, 1)
        self.assertIn("controlled split check failed", str(service.state.get("error", "")))

    def test_training_command_uses_frozen_exit_set_and_runtime_bound(self):
        command = service.build_training_command(
            Path("/uploaded/repo"), Path("/data/run"), "full", (12, 16, 20),
            max_epochs=2, max_runtime_seconds=14400,
        )
        self.assertEqual(command[2], "/uploaded/repo/retrain/16_adas1_multiexit_4b/scripts/train_m16.py")
        self.assertIn(("--stage", "full"), list(zip(command, command[1:])))
        self.assertIn(("--exit-layers", "12,16,20"), list(zip(command, command[1:])))
        self.assertIn(("--max-epochs", "2"), list(zip(command, command[1:])))
        self.assertIn(("--max-runtime-seconds", "14400"), list(zip(command, command[1:])))
        with self.assertRaisesRegex(ValueError, "unsupported M16 training stage"):
            service.build_training_command(Path("/uploaded/repo"), Path("/data/run"), "probe")

    def test_training_artifact_selection_requires_and_preserves_both_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "run"
            (output / "teacher").mkdir(parents=True)
            (output / "multiexit").mkdir(parents=True)
            (output / "teacher/teacher_head.pt").write_bytes(b"teacher")
            (output / "multiexit/exit_heads.pt").write_bytes(b"exits")
            (output / "run_report.json").write_text("{}", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            selected = service.training_artifact_sources(output, manifest, "full")
            names = {name.as_posix() for _, name in selected}
            self.assertIn("training/full/teacher/teacher_head.pt", names)
            self.assertIn("training/full/multiexit/exit_heads.pt", names)
            self.assertIn("input/data/controlled/manifest.json", names)

    def test_vk_start_defaults_to_full_training_with_frozen_intermediate_exits(self):
        script = (Path(__file__).resolve().parent / "vkong_start.sh").read_text(encoding="utf-8")
        self.assertIn('export M16_SERVICE_STAGE="${M16_SERVICE_STAGE:-full}"', script)
        self.assertIn('export M16_EXIT_LAYERS="${M16_EXIT_LAYERS:-12,16,20}"', script)

    def test_artifact_selection_and_package_names_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "run"
            output.mkdir()
            for name in ("probe_report.json", "exit_curve.csv", "exit_curve.svg"):
                (output / name).write_text("ok", encoding="utf-8")
            for layer in service.configured_layers():
                layer_dir = output / f"layer_{layer}"
                layer_dir.mkdir()
                (layer_dir / "probe.pt").write_bytes(b"weights")
                (layer_dir / "history.json").write_text("[]", encoding="utf-8")
            feature_dir = output / "features"
            feature_dir.mkdir()
            (feature_dir / "train_features.pt").write_bytes(b"cache")
            input_manifest = root / "controlled_manifest.json"
            input_manifest.write_text("{}", encoding="utf-8")

            selected = service.artifact_sources(output, input_manifest, service.configured_layers())
            selected_names = {name.as_posix() for _, name in selected}
            self.assertEqual(len([name for name in selected_names if name.endswith("/probe.pt")]), 8)
            self.assertIn("input/data/controlled/manifest.json", selected_names)
            self.assertNotIn("features/train_features.pt", selected_names)
            self.assertEqual(service.sha256_file(output / "layer_4/probe.pt"), hashlib.sha256(b"weights").hexdigest())
            self.assertTrue(service.ARCHIVE.name.endswith("-results.tar.gz"))

    def test_existing_output_directory_gets_a_fresh_run_id(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime_root = Path(temp)
            previous_run_id = service.RUN_ID
            service.RUN_ID = "collision"
            try:
                (runtime_root / "collision").mkdir()
                layers = service.configured_layers()
                code = (
                    "import os; from pathlib import Path; "
                    "root=Path(os.environ['M16_OUTPUT']); root.mkdir(parents=True, exist_ok=False); "
                    "[(root.joinpath(name).write_text('ok')) for name in "
                    "('probe_report.json','exit_curve.csv','exit_curve.svg')]; "
                    f"layers={layers!r}; "
                    "[(root.joinpath(f'layer_{n}').mkdir(), "
                    "root.joinpath(f'layer_{n}/probe.pt').write_bytes(b'weights'), "
                    "root.joinpath(f'layer_{n}/history.json').write_text('[]')) for n in layers]"
                )
                service.state.clear()
                service.state.update({"status": "starting", "run_id": service.RUN_ID})
                exit_code = service.train_and_package(
                    [sys.executable, "-c", code], service.WORKSPACE_ROOT, runtime_root
                )
                self.assertEqual(exit_code, 0, service.state.get("error"))
                self.assertNotEqual(service.state["run_id"], "collision")
            finally:
                service.RUN_ID = previous_run_id

    def test_successful_service_package_has_named_hashed_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime_root = Path(temp)
            layers = service.configured_layers()
            code = (
                "import json, os; from pathlib import Path; "
                "root=Path(os.environ['M16_OUTPUT']); root.mkdir(parents=True, exist_ok=False); "
                "[(root.joinpath(name).write_text('ok')) for name in "
                "('probe_report.json','exit_curve.csv','exit_curve.svg')]; "
                f"layers={layers!r}; "
                "[(root.joinpath(f'layer_{n}').mkdir(), "
                "root.joinpath(f'layer_{n}/probe.pt').write_bytes(b'weights'), "
                "root.joinpath(f'layer_{n}/history.json').write_text('[]')) for n in layers]"
            )
            service.state.clear()
            service.state.update({"status": "starting", "run_id": service.RUN_ID})
            code_status = service.train_and_package(
                [sys.executable, "-c", code], service.WORKSPACE_ROOT, runtime_root
            )
            self.assertEqual(code_status, 0)
            archive_path = Path(str(service.state["archive_path"]))
            self.assertTrue(archive_path.name.endswith("-results.tar.gz"))
            self.assertEqual(service.state["sha256"], service.sha256_file(archive_path))
            with tarfile.open(archive_path, "r:gz") as archive:
                members = set(archive.getnames())
                manifest_member = next(name for name in members if name.endswith("/artifact_manifest.json"))
                manifest = json.load(archive.extractfile(manifest_member))
                self.assertEqual(len(manifest["artifacts"]), 20)
                self.assertTrue(any(name.endswith("/input/data/controlled/manifest.json") for name in members))
                self.assertFalse(any("/features/" in name for name in members))
                first_entry = manifest["artifacts"][0]
                payload = archive.extractfile(first_entry["path"]).read()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), first_entry["sha256"])

    def test_streams_child_output_byte_for_byte(self):
        expected = b"batch 1 loss=0.25\rwarning\n"
        code = "import os; os.write(1,b'batch 1 loss=0.25\\r'); os.write(2,b'warning\\n')"
        with tempfile.TemporaryFile() as log:
            app_output = io.BytesIO()
            process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, bufsize=0)
            exit_code = service.stream_process_output(process, log, app_output)
            log.seek(0)
            self.assertEqual(exit_code, 0)
            self.assertEqual(log.read(), expected)
            self.assertEqual(app_output.getvalue(), expected)

    def test_streams_child_output_respects_hard_timeout(self):
        code = "import os,time,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); os.write(1,b'child started\\n'); time.sleep(30)"
        with tempfile.TemporaryFile() as log:
            app_output = io.BytesIO()
            process = subprocess.Popen(
                [sys.executable, "-c", code], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=0, start_new_session=True,
            )
            started_at = time.monotonic()
            exit_code = service.stream_process_output(
                process, log, app_output, timeout_seconds=0.5
            )
            elapsed = time.monotonic() - started_at
            log.seek(0)
            self.assertEqual(exit_code, 124)
            self.assertIsNotNone(process.poll())
            self.assertEqual(log.read(), b"child started\n")
            self.assertEqual(app_output.getvalue(), b"child started\n")
            self.assertLess(elapsed, 5.0)

    def test_default_failure_grace_keeps_remote_status_available(self):
        self.assertGreaterEqual(service.FAILURE_GRACE_SECONDS, 30.0)

    def test_failing_probe_is_visible_over_http_and_service_exits_nonzero(self):
        old_grace = service.FAILURE_GRACE_SECONDS
        service.FAILURE_GRACE_SECONDS = 0.4
        with tempfile.TemporaryDirectory() as temp:
            service.state.clear()
            service.state.update({"status": "starting", "run_id": service.RUN_ID})
            server = ThreadingHTTPServer(("127.0.0.1", 0), service.Handler)
            result = {}
            command = [sys.executable, "-c",
                       "import sys; print('synthetic probe failure', file=sys.stderr); sys.exit(23)"]
            worker = threading.Thread(target=lambda: result.setdefault(
                "exit_code", service.serve(server, command, service.WORKSPACE_ROOT, Path(temp))))
            worker.start()
            url = f"http://127.0.0.1:{server.server_address[1]}"
            deadline = time.monotonic() + 5
            observed = None
            while time.monotonic() < deadline:
                try:
                    with urlopen(url + "/status", timeout=0.5) as response:
                        observed = json.loads(response.read())
                    if observed.get("status") == "failed":
                        break
                except OSError:
                    pass
                time.sleep(0.03)
            with urlopen(url + "/logs", timeout=1) as response:
                logs = response.read()
            worker.join(timeout=5)
            server.server_close()
            service.FAILURE_GRACE_SECONDS = old_grace

            self.assertFalse(worker.is_alive(), "service did not terminate after failed probe")
            self.assertEqual(result.get("exit_code"), 23)
            self.assertEqual(observed["status"], "failed")
            self.assertEqual(observed["exit_code"], 23)
            self.assertIn(b"synthetic probe failure", logs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
