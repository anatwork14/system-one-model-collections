#!/usr/bin/env python3
"""Fetch pinned source/model artifacts and update provenance locks.

Run with Python 3.10+ after installing environment/base_requirements.txt.
Existing checkouts/downloads are validated and never silently updated.
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ required; use uv run --python 3.12 tools/fetch_all.py")
try:
    import yaml
    from huggingface_hub import HfApi, snapshot_download
except ImportError as exc:
    raise SystemExit(f"Missing setup dependency: {exc}. Install environment/base_requirements.txt")

NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")
SOURCES = [
    ("released/00_semif/upstream", "https://github.com/iwillcodeu/openjev.git", "master"),
    ("released/01_simple_jev/upstream", "https://github.com/featherless-ai/simple-jev.git", "main"),
    ("released/02_tev1_4b/upstream", "https://github.com/togethercomputer/tev1.git", "main"),
    ("released/03_jevk5_4b/upstream", "https://github.com/allebee/jevk5.git", "main"),
    ("released/04_kev_4b/upstream", "https://github.com/jaredpalmer/kev.git", "main"),
    ("retrain/10_nimble_4b/upstream", "https://github.com/bespokelabsai/nimble.git", "main"),
    ("retrain/11_clm_style_4b/upstream", "https://github.com/Contrastive-LM/CLM.git", "main"),
]
MODELS = [
    ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "shared/models/qwen3.5-4b"),
    ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b", "shared/models/qwen3.5-4b-base"),
    ("togethercomputer/Tev1-4B-experimental", None, "released/02_tev1_4b/weights/model"),
    ("alibiserikbay/JevK5", None, "released/03_jevk5_4b/weights/model"),
    ("jaredpalmer/kev-4b", None, "released/04_kev_4b/weights/released"),
]

def run(args, cwd=None):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(args, cwd=cwd, check=True)

def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def tree_manifest(path):
    out = []
    for f in sorted(path.rglob("*")):
        if f.is_file() and ".cache" not in f.parts:
            out.append({"path": f.relative_to(path).as_posix(), "size_bytes": f.stat().st_size, "sha256": digest_file(f)})
    return out

def existing_marker_matches(path, sha):
    marker = path / ".workspace-revision"
    return marker.exists() and marker.read_text().strip() == sha

def checkout_sources():
    resolved = {}
    for rel, url, ref in SOURCES:
        dest = ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        marker = dest / "UPSTREAM_README.txt"
        if marker.exists():
            marker.unlink()
        if not (dest / ".git").exists():
            if dest.exists() and any(dest.iterdir()):
                raise RuntimeError(f"Refusing to replace non-empty upstream path: {dest}")
            run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(dest)])
            run(["git", "fetch", "--depth", "1", "origin", ref], cwd=dest)
            run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=dest)
        if not (dest / "README.md").exists():
            # Recover a fresh --no-checkout clone from an interrupted fetch.
            # These repositories all have a root README; upstream trees are otherwise read-only.
            run(["git", "checkout", "--detach", "-f", "HEAD"], cwd=dest)
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()
        remotes = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=dest, text=True).strip()
        if remotes.rstrip("/") != url.removesuffix(".git").rstrip("/") and remotes != url:
            raise RuntimeError(f"Origin mismatch in {dest}: {remotes}")
        resolved[rel] = {"repo": url, "requested_ref": ref, "commit": sha}
        print(f"source pinned: {rel} @ {sha}", flush=True)
    # Kev source is shared for local Kev-derived ablations; no duplicate checkout.
    for rel in ("retrain/12_pointer_head_only_4b/upstream", "retrain/13_kev_fullft_4b/upstream"):
        link = ROOT / rel
        target = os.path.relpath(ROOT / "released/04_kev_4b/upstream", link.parent)
        if link.is_symlink() and os.readlink(link) != target:
            link.unlink()
        if not link.exists():
            if link.is_dir():
                shutil.rmtree(link)
            link.symlink_to(target, target_is_directory=True)
        resolved[rel] = {"shared_source": "released/04_kev_4b/upstream", "commit": resolved["released/04_kev_4b/upstream"]["commit"]}
    return resolved

def download_models():
    api = HfApi()
    resolved = {}
    for repo, pinned, rel in MODELS:
        dest = ROOT / rel
        info = api.model_info(repo, revision=pinned) if pinned else api.model_info(repo)
        sha = info.sha
        if dest.exists() and any(dest.iterdir()):
            existing = dest / ".workspace-revision"
            if existing.exists() and existing.read_text().strip() != sha:
                raise RuntimeError(f"Existing model revision mismatch at {dest}; preserving existing files")
            if not existing.exists() and not (dest / ".cache/huggingface/download").exists():
                raise RuntimeError(f"Existing model folder lacks .workspace-revision: {dest}; inspect before continuing")
        if not dest.exists() or not existing_marker_matches(dest, sha):
            dest.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=repo, revision=sha, local_dir=str(dest))
            (dest / ".workspace-revision").write_text(sha + "\n", encoding="utf-8")
        manifest = tree_manifest(dest)
        resolved[repo] = {"revision": sha, "path": rel, "files": manifest}
        print(f"model pinned: {repo} @ {sha}; files={len(manifest)}", flush=True)
    return resolved

def update_locks(sources, models):
    map_source = {
        "released/00_semif": "released/00_semif/upstream",
        "released/01_simple_jev": "released/01_simple_jev/upstream",
        "released/02_tev1_4b": "released/02_tev1_4b/upstream",
        "released/03_jevk5_4b": "released/03_jevk5_4b/upstream",
        "released/04_kev_4b": "released/04_kev_4b/upstream",
        "retrain/10_nimble_4b": "retrain/10_nimble_4b/upstream",
        "retrain/11_clm_style_4b": "retrain/11_clm_style_4b/upstream",
        "retrain/12_pointer_head_only_4b": "retrain/12_pointer_head_only_4b/upstream",
        "retrain/13_kev_fullft_4b": "retrain/13_kev_fullft_4b/upstream",
    }
    hf_map = {"released/02_tev1_4b": "togethercomputer/Tev1-4B-experimental",
              "released/03_jevk5_4b": "alibiserikbay/JevK5",
              "released/04_kev_4b": "jaredpalmer/kev-4b"}
    for method, source_rel in map_source.items():
        lock_path = ROOT / method / "ARTIFACTS.lock"
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        lock["source"]["commit"] = sources[source_rel]["commit"]
        lock["source"]["resolved_ref"] = sources[source_rel].get("requested_ref", sources[source_rel].get("shared_source"))
        lock["downloaded_at"] = NOW
        if method in hf_map:
            repo = hf_map[method]
            lock["checkpoint"]["revision"] = models[repo]["revision"]
            lock["checkpoint"]["files"] = [{"path": x["path"], "size_bytes": x["size_bytes"], "sha256": x["sha256"]} for x in models[repo]["files"]]
        data_manifest = ROOT / method / lock.get("data", {}).get("manifest", "")
        if data_manifest.is_file():
            data_dir = ROOT / method / "data"
            lock["data"]["hashes"] = {
                "manifest_sha256": digest_file(data_manifest),
                "files": [{"path": x["path"], "size_bytes": x["size_bytes"], "sha256": x["sha256"]}
                          for x in tree_manifest(data_dir) if x["path"].startswith(("raw/", "original_recipe/", "processed/", "splits/", "manifests/"))],
            }
        lock_path.write_text(yaml.safe_dump(lock, sort_keys=False, allow_unicode=True), encoding="utf-8")
    models_lock_path = ROOT / "shared/models/MODELS.lock"
    ml = yaml.safe_load(models_lock_path.read_text(encoding="utf-8"))
    for key, repo in (("qwen35_4b", "Qwen/Qwen3.5-4B"), ("qwen35_4b_base", "Qwen/Qwen3.5-4B-Base")):
        ml["models"][key]["revision"] = models[repo]["revision"]
        ml["models"][key]["files"] = [{"path": x["path"], "size_bytes": x["size_bytes"], "sha256": x["sha256"]} for x in models[repo]["files"]]
    ml["downloaded_at"] = NOW
    ml["file_manifest"] = "embedded per-model SHA256 manifest"
    models_lock_path.write_text(yaml.safe_dump(ml, sort_keys=False, allow_unicode=True), encoding="utf-8")

def prepare_data():
    src = ROOT / "retrain/10_nimble_4b/upstream/data"
    canonical = ROOT / "shared/datasets/nimble/original-2676"
    canonical.mkdir(parents=True, exist_ok=True)
    names = ("train.jsonl", "eval.jsonl", "manifest.json")
    for name in names:
        source = src / name
        if not source.exists():
            raise RuntimeError(f"Expected Nimble source file missing: {source}")
        target = canonical / name
        if not target.exists():
            shutil.copy2(source, target)
    local_raw = ROOT / "retrain/10_nimble_4b/data/raw"
    for name in names:
        dst = local_raw / name
        srcfile = canonical / name
        if dst.exists() and digest_file(dst) != digest_file(srcfile):
            raise RuntimeError(f"Existing local Nimble file differs from canonical copy: {dst}")
        if not dst.exists():
            shutil.copy2(srcfile, dst)
    for name in names:
        shutil.copy2(canonical / name, ROOT / "shared/datasets/nimble" / name)
    rows = {name: sum(1 for line in (canonical / name).open(encoding="utf-8") if line.strip()) if name.endswith(".jsonl") else None for name in names}
    manifest = {"status": "downloaded", "source": "bespokelabsai/nimble", "source_ref": "main (exact commit in method ARTIFACTS.lock)", "counts": rows, "files": tree_manifest(canonical), "label_origin": "synthetic/model-checked; not human-reviewed", "evaluation_use": "frozen holdout; never used for training/model selection"}
    (ROOT / "retrain/10_nimble_4b/data/manifests/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (ROOT / "shared/datasets/nimble/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Nimble data copied: {rows}", flush=True)

def create_links():
    for method, model_rel in [
        ("released/00_semif", "shared/models/qwen3.5-4b"),
        ("released/01_simple_jev", "shared/models/qwen3.5-4b"),
        ("released/02_tev1_4b", "shared/models/qwen3.5-4b"),
        ("released/03_jevk5_4b", "shared/models/qwen3.5-4b"),
        ("released/04_kev_4b", "shared/models/qwen3.5-4b-base"),
        ("retrain/10_nimble_4b", "shared/models/qwen3.5-4b"),
        ("retrain/11_clm_style_4b", "shared/models/qwen3.5-4b"),
        ("retrain/12_pointer_head_only_4b", "shared/models/qwen3.5-4b-base"),
        ("retrain/13_kev_fullft_4b", "shared/models/qwen3.5-4b-base"),
    ]:
        link = ROOT / method / "weights/base_model"
        target = os.path.relpath(ROOT / model_rel, link.parent)
        if link.is_symlink():
            if os.readlink(link) != target:
                link.unlink(); link.symlink_to(target, target_is_directory=True)
        elif link.exists():
            raise RuntimeError(f"Refusing to replace existing weights/base_model: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)

def create_hub_cache_aliases(models):
    """Expose the single local base snapshots through the HF cache resolver too."""
    hub = ROOT / ".hf-home/hub"
    for repo in ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B-Base"):
        model = models[repo]
        cache_repo = hub / ("models--" + repo.replace("/", "--"))
        snapshots = cache_repo / "snapshots"
        snapshots.mkdir(parents=True, exist_ok=True)
        alias = snapshots / model["revision"]
        target = ROOT / model["path"]
        if alias.is_symlink():
            if alias.resolve() != target.resolve():
                alias.unlink()
                alias.symlink_to(os.path.relpath(target, snapshots), target_is_directory=True)
        elif not alias.exists():
            alias.symlink_to(os.path.relpath(target, snapshots), target_is_directory=True)
        refs = cache_repo / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text(model["revision"] + "\n", encoding="utf-8")

def create_artifact_links():
    """Expose the Kev adapter and head in their documented method-local slots without copies."""
    released = ROOT / "released/04_kev_4b/weights/released"
    for name, target_dir in (("adapter_model.safetensors", "adapter"), ("adapter_config.json", "adapter"), ("head.pt", "head")):
        src = released / name
        if not src.is_file():
            raise RuntimeError(f"Expected Kev release artifact missing: {src}")
        dest_dir = ROOT / "released/04_kev_4b/weights" / target_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        relative = os.path.relpath(src, dest_dir)
        if dest.is_symlink() and os.readlink(dest) != relative:
            dest.unlink()
        if not dest.exists():
            dest.symlink_to(relative)

def main():
    if os.environ.get("FETCH_CONFIRM") != "yes":
        raise SystemExit("Set FETCH_CONFIRM=yes to fetch repositories and large model weights.")
    sources = checkout_sources()
    models = download_models()
    prepare_data()
    create_links()
    create_artifact_links()
    create_hub_cache_aliases(models)
    update_locks(sources, models)
    state = {"downloaded_at": NOW, "sources": sources, "models": {k: {"revision": v["revision"], "path": v["path"], "file_count": len(v["files"])} for k, v in models.items()}, "status": "all_fetches_completed"}
    (ROOT / "DOWNLOAD_STATUS.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    (ROOT / "DOWNLOAD_STATUS.md").write_text("# Download status\n\nAll planned sources and model snapshots were fetched successfully. Exact source commits, Hub revisions, file sizes, and SHA-256 hashes are recorded in the adjacent lock files and `DOWNLOAD_STATUS.json`.\n\nDownloaded at: " + NOW + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
