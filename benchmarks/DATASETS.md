# Dataset inventory and handling

The benchmark suite defines 14 evaluation entries backed by 12 unique raw datasets. `banking77_bolt` and `clinc150_bolt` are protocol overlays that point to the System-One raw copy. The overlay folders contain no `raw/` directory.

## Unique raw sources

| ID | Dataset | Expected examples | Classes / outputs | Storage owner |
|---|---|---:|---:|---|
| S0 | JevBench Public | 231 | Variable native outputs | `system_one/00_jevbench` |
| S1 | BANKING77 | 13,083 | 77 | `system_one/01_banking77` |
| S2 | CLINC150 + OOS | 23,700 | 150 + OOS | `system_one/02_clinc150_oos` |
| S3 | BoolQ | 12,697 labeled | 2 | `system_one/03_boolq` |
| S4 | GoEmotions | 58,011 distinct IDs; 211,225 annotation rows | 27 + Neutral | `system_one/04_goemotions` |
| S5 | When2Call | 3,652 eval projection | 4 | `system_one/05_when2call` |
| S6 | BFCL | Selected subset; pinned projection defines count | Dynamic tools | `system_one/06_bfcl` |
| O2 | HWU64 via BOLT | 9,677 processed projection | 64 | `openworld/12_hwu64_bolt` |
| O3 | StackOverflow via BOLT | 19,985 processed projection | 20 | `openworld/13_stackoverflow_bolt` |
| V0 | CIFAR-10 | 60,000 | 10 | `vision/20_cifar10` |
| V1 | Oxford-IIIT Pets | 7,349 | 37 | `vision/21_oxford_pets` |
| V2 | Oxford Flowers-102 | 8,189 | 102 | `vision/22_flowers102` |

The raw sources are pinned and downloaded. Observed split counts and raw SHA-256 values are recorded in `manifests/download_report.json`, each `SOURCE.lock.yaml`, and the checksum files. GoEmotions' pinned source has two more distinct IDs than the planning estimate; its annotation-row count and benchmark filtered splits were cross-checked.

Oxford Pets and Flowers-102 were transferred from pinned Hugging Face mirrors after the official Oxford host proved too slow to complete reliably. The original Oxford source URLs and mirror revisions are both recorded in their source locks. Check upstream use and redistribution terms before sharing downloaded files.

## Deliberately deferred

AG News, Food-101, ImageNet-1K, POPE, MME, the full Decision Index, medical/legal datasets, and the full BOLT catalog are out of the initial milestone.

## Data handling rules

- Never download a second BANKING77 or CLINC copy for BOLT.
- Do not redistribute raw data unless its upstream terms explicitly allow it.
- Do not use evaluation splits for training or model selection unless an experiment clearly declares in-domain training.
- Store source files under one owning dataset's `raw/`; overlays store `raw_ref.yaml` and derived protocols only.
- Keep all source revisions immutable and all transformations versioned.
