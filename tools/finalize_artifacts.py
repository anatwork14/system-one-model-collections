#!/usr/bin/env python3
"""Rebuild local SHA-256 provenance locks without contacting external services."""
import json
from datetime import datetime, timezone
from pathlib import Path

from fetch_all import ROOT, tree_manifest, update_locks

state_path = ROOT / "DOWNLOAD_STATUS.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
sources = state["sources"]

# Record frozen split membership separately from row order and source bytes.
nimble = ROOT / "retrain/10_nimble_4b/data"
for split, filename in (("train", "train.jsonl"), ("frozen_eval", "eval.jsonl")):
    source = nimble / "raw" / filename
    ids = [json.loads(line)["id"] for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    (nimble / "splits" / f"{split}_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

# Capture exact Tev1 recipe output and its split counts once the source build is present.
tev = ROOT / "released/02_tev1_4b/data"
tev_root = tev / "original_recipe/new-v1"
if (tev_root / "manifest.json").is_file():
    source_manifest = json.loads((tev_root / "manifest.json").read_text(encoding="utf-8"))
    source_lock = tev / "build_src/sources.lock.json"
    if not source_lock.is_file():
        raise SystemExit(f"Missing pinned Tev1 dataset source lock: {source_lock}")
    local_source_lock = tev / "manifests/tev1_sources.lock.json"
    local_source_lock.parent.mkdir(parents=True, exist_ok=True)
    local_source_lock.write_bytes(source_lock.read_bytes())
    counts = {split: int(info["count"]) for split, info in source_manifest["splits"].items()}
    dataset_manifest = {
        "status": "built_and_validated",
        "recipe": "Tev1 new-v1, exact pinned source commit",
        "source_commit": sources["released/02_tev1_4b/upstream"]["commit"],
        "source_datasets_lock": "manifests/tev1_sources.lock.json",
        "source_datasets_lock_sha256": __import__("fetch_all").digest_file(local_source_lock),
        "source_datasets": json.loads(local_source_lock.read_text(encoding="utf-8")),
        "counts": counts,
        "files": tree_manifest(tev_root),
        "upstream_build_manifest": source_manifest,
        "license_notes": "See upstream/DATA_SOURCES.md and build_src/sources.lock.json; some source license metadata is unknown or unspecified.",
    }
    (tev / "manifests").mkdir(parents=True, exist_ok=True)
    (tev / "manifests/manifest.json").write_text(json.dumps(dataset_manifest, indent=2) + "\n", encoding="utf-8")

models = {}
for repo, row in state["models"].items():
    path = ROOT / row["path"]
    revision_file = path / ".workspace-revision"
    if not revision_file.is_file():
        raise SystemExit(f"Missing immutable revision marker: {revision_file}")
    revision = revision_file.read_text(encoding="utf-8").strip()
    models[repo] = {"revision": revision, "path": row["path"], "files": tree_manifest(path)}
    if len(models[repo]["files"]) != row["file_count"]:
        raise SystemExit(f"File count changed for {repo}: {len(models[repo]['files'])} != {row['file_count']}")
update_locks(sources, models)
state["datasets"] = {
    "nimble": {"path": "shared/datasets/nimble", "train": 2676, "frozen_eval": 324,
               "manifest_sha256": __import__("fetch_all").digest_file(ROOT / "retrain/10_nimble_4b/data/manifests/manifest.json")},
}
tev_manifest = ROOT / "released/02_tev1_4b/data/manifests/manifest.json"
if tev_manifest.is_file():
    state["datasets"]["tev1_new_v1"] = {"path": "released/02_tev1_4b/data/original_recipe/new-v1",
                                         "counts": counts,
                                         "manifest_sha256": __import__("fetch_all").digest_file(tev_manifest)}
state["downloaded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
state["status"] = "sources_models_and_initial_datasets_prepared"
state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
(ROOT / "DOWNLOAD_STATUS.md").write_text(
    "# Download status\n\n"
    "Pinned source snapshots, shared base checkpoints, and all five published released checkpoints are present. "
    "Nimble's 2,676 train / 324 frozen evaluation records and Tev1's published `new-v1` train/development recipe "
    "are materialized with content hashes. Exact source commits, Hub revisions, file hashes, and data manifests "
    "are recorded in the adjacent locks and `DOWNLOAD_STATUS.json`.\n\n"
    "Training, model inference, target-host resource validation, and performance measurement have not been run.\n\n"
    f"Prepared at: {state['downloaded_at']}\n", encoding="utf-8")
print("Artifact locks refreshed from local pinned files; no network access used.")
