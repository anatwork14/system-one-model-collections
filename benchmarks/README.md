# Minimal JEV / Open-World / VLM Benchmark Suite

This directory defines a common evaluation suite for Qwen3.5-4B → System-One methods. It is designed to cover native JEV decisions, closed-set classification, open-set and runtime-supplied unseen classes, tool routing, and image classification at 10, 37, and 102 classes.

The suite defines 14 benchmark entries targeting 12 unique raw datasets. BOLT Banking77 and CLINC are protocol overlays over the two System-One raw copies. Raw files are not checked into source control and downloads are never overwritten silently.

## Layout

```text
benchmarks/
├── DATASETS.md
├── registry.yaml
├── checksums/
├── sources/                 # pinned upstream source checkouts
├── system_one/              # seven canonical JEV/System-One datasets
├── openworld/               # four BOLT protocols; two reference raw data
├── vision/                  # CIFAR-10, Oxford Pets, Flowers-102
├── scripts/
└── manifests/
```

Every data owner has `DATASET.md`, `SOURCE.lock.yaml`, `checksums.sha256`, and `raw/`, `processed/`, `splits/`, and `scripts/` directories. BOLT overlays have `raw_ref.yaml` instead of a `raw/` directory. Per-dataset scripts delegate to shared tooling.

## Initial setup

```bash
python benchmarks/scripts/bootstrap_layout.py
python benchmarks/scripts/download_core.py --list
```

The source revisions are pinned in each `SOURCE.lock.yaml`. The core raw data and selected protocol inputs have been downloaded to this workspace; raw payloads are excluded from Git.

After revisions and source adapters are recorded, use:

```bash
python benchmarks/scripts/download_core.py --group system_one
python benchmarks/scripts/download_core.py --group openworld
python benchmarks/scripts/download_core.py --group vision
python benchmarks/scripts/validate_downloads.py
python benchmarks/scripts/prepare_core.py --group all
python benchmarks/scripts/checksum_all.py
```

To download one source, use `--dataset banking77`. Re-running a completed download does not overwrite existing files; use the documented per-dataset preparation command for derived outputs.

## Shared decision schema

Text and vision examples use the same JSONL envelope:

```json
{"id":"dataset/split/row","state":{"text":"..."},"question":{"id":"decision","type":"choice","instructions":"Choose the best matching class.","criteria":{"class_a":"Description A","class_b":"Description B"}},"target":"class_a","metadata":{"dataset":"example","split":"test"}}
```

`state.image` may contain a path relative to the owning dataset for vision records. BoolQ uses `question.type: "noul"`. GoEmotions produces 28 Noul decisions per text, with a shared source ID. Dataset-specific cards define target semantics and metrics.
The machine-readable contract is in [`decision_record.schema.json`](decision_record.schema.json).

## Open-world protocols

Materialize the BOLT `known75` fold first. Derive three evaluation views: OW-0 closed known classes; OW-1 known candidates plus `UNKNOWN`; OW-2 known candidates plus an unseen class description and `UNKNOWN`, with the unseen class as target. Keep unseen classes absent from all training class lists. Native CLINC OOS remains a separate result from BOLT-held-out CLINC intents.

## Status and reproducibility

`registry.yaml` is the dataset index; `manifests/download_report.json` records observed source split counts and SHA-256 verification. All 14 entries (12 unique raw dataset sources) pass download validation and the normalized JSONL and open-world views have been generated. Per-dataset `status.json` files record preparation counts, and `manifests/prepare_report.json` summarizes normalized record totals and validation status.

Oxford Pets and Flowers-102 payloads came from pinned Hugging Face transfer mirrors because the official Oxford host was too slow to finish reliably. Their official source URLs, mirror revisions, and payload hashes are recorded in their locks. GoEmotions has 58,011 distinct IDs in the pinned raw CSVs (211,225 annotation rows), two more than the 58,009 planning figure; the filtered split counts match the plan exactly.

Vision `state.image` values point into the owning raw Parquet file as `raw/...parquet#row=N`. This keeps one copy of image bytes; consumers should read that Parquet row rather than expecting an extracted image file. BOLT's known75 files define the class partition. For the two overlay datasets, evaluation rows are derived from the canonical raw source and are not a byte-for-byte copy of BOLT's row sampling.
