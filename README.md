# System-One Qwen3.5-4B Model Zoo

An experimental workspace for comparing ways to turn Qwen3.5-4B into a typed decision model. Upstream source trees, released checkpoints, reproductions, and local evaluation outputs have separate locations and identities.

## Status

The pinned source snapshots, both Qwen 4B base models, all five released 4B checkpoints, the complete pinned Cloudflare Clef-Flash 9B release, and the Nimble/Tev1 datasets are present. Clef-Flash lives alongside the other releases under `released/05_clef_flash_9b/`; its 4B reproduction is under `retrain/15_clef_style_4b/`. The official revision, verification, and file hashes are recorded in the Clef-Flash lock and manifest.

## Layout

- `shared/models/`: single copies of the two pinned Qwen bases.
- `released/`: five public 4B implementations/checkpoints and the official Cloudflare Clef-Flash 9B release.
- `retrain/`: 4B reproduction recipes and controlled ablations, including our Clef-style Qwen3.5-4B arm.
- `shared/datasets/`: canonical shared datasets and protocols.
- `benchmarks/`: JEV, open-world, tool-routing, and VLM benchmark definitions with pin-gated download tooling (12 unique raw datasets planned).
- `evaluation/`: adapters, metrics, and common evaluation runner.
- `results/`: released smoke checks and controlled experiment tables.
- `environment/`: Python environment specification and diagnostics.

## Bootstrap

Use Python 3.12+, Git with Git LFS, and network access for the download script. Upstream projects have incompatible Torch pins, so the root environment is for shared tooling; see `environment/ENVIRONMENTS.md` for method-specific runtime environments.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r environment/base_requirements.txt
python -m pip install -U uv
bash tools/fetch_all.sh
```

Review every repository's `ARTIFACTS.lock` after fetching. The fetch script checks out exact source commits and Hugging Face revisions and records SHA-256 file manifests. It never edits upstream files. `weights/base_model` entries are relative symlinks to `shared/models/`.

Prepare the data recipes after fetching with:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" bash released/02_tev1_4b/scripts/build_data.sh
bash retrain/10_nimble_4b/scripts/prepare.sh
bash retrain/11_clm_style_4b/scripts/prepare_data.sh
```

The first command rebuilds Tev1's pinned `new-v1` data in method-local staging. The next two validate/copy the Nimble-controlled split and derive its Kev/CLM input formats. These commands do not train models.

## First commands

```bash
python3 environment/gpu_check.py       # optional diagnostic; does not train
python3 evaluation/run_all.py --list
python3 benchmarks/scripts/download_core.py --list
```

Released smoke inference requires the relevant model weights and an available inference runtime. Results are recorded separately from benchmarks. Retraining configs and scripts are prepared but are intentionally not run here.

## Scientific identity

`UPSTREAM_*` result IDs denote released artifacts. `OURS_*` IDs denote local reproductions/ablations. Do not merge these identities in result tables. See `results/README.md` for the controlled comparison protocol.

## Source notes

The workspace follows the published descriptions of [SemIf](https://github.com/iwillcodeu/openjev), [Simple Jev](https://github.com/featherless-ai/simple-jev), [Tev1](https://github.com/togethercomputer/tev1), [JevK5](https://github.com/allebee/jevk5), [Kev](https://github.com/jaredpalmer/kev), [Nimble](https://github.com/bespokelabsai/nimble), and [CLM](https://github.com/Contrastive-LM/CLM). The per-method files state which details are source-verified and which are planned reproduction choices.
