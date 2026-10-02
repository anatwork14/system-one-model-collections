#!/usr/bin/env python3
"""Create the documented per-dataset benchmark scaffold from registry.yaml."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

DETAILS = {
    "jevbench_public": ("JevBench Public", "System-One native decisions", "Evaluate Choice, Score, Noul, routing, policy, numeric, probability, ambiguity, and multi-hop behavior. Only the 231 public examples are in scope; 303 tasks are private or held out.", "Native decision record", "evaluation only; no training", "231 public (72 original, 48 easy, 111 hard)", "Variable native outputs", "Pin the repository commit, then copy only public task files into raw/.", "source-specific native evaluator"),
    "banking77": ("BANKING77", "System-One and open-world intent classification", "Measure fine-grained 77-way intent classification and candidate routing; the same raw data is referenced by the BOLT overlay.", "Customer utterance", "Choice(77), label name only and label plus description variants", "10,003 train; 3,080 test; 13,083 total", "77 intents", "Keep official train and test assignments; never duplicate the raw files for BOLT.", "accuracy, macro-F1, NLL, Brier, ECE"),
    "clinc150_oos": ("CLINC150 + OOS", "System-One classification and native out-of-scope detection", "Measure 150 intent classes, native OOS detection, abstention, and high-cardinality choices. Native OOS and BOLT-held-out intents are distinct protocols.", "User utterance", "Choice(150); OOS is a separate unknown target", "23,700 labeled records: 150 in-scope classes plus OOS", "150 intents plus OOS", "Use data_full.json only; preserve official train, validation, and test splits.", "accuracy, macro-F1, OOS AUROC/AUPR, NLL, Brier, ECE"),
    "boolq": ("BoolQ", "System-One binary semantic decision", "Evaluate passage-grounded yes/no decisions through the Noul interface.", "Passage and question", "Noul: yes/no probability", "9,427 train; 3,270 dev; 12,697 labeled total", "Boolean yes/no", "Use labeled train and dev only. Exclude the unlabeled test set.", "accuracy, NLL, Brier, ECE"),
    "goemotions": ("GoEmotions", "System-One multi-label and fan-out decisions", "Evaluate 28 independent emotion decisions per text, including Neutral, and compare independent scoring with joint schema heads.", "Reddit comment", "28 Noul questions per example", "58,009 raw records; expected filtered train/dev/test: 43,410/5,426/5,427", "27 emotions plus Neutral", "Use the filtered split; retain multi-label targets and document label mapping.", "micro/macro-F1, per-label AUROC, NLL/Brier/ECE where defined"),
    "when2call": ("When2Call", "System-One tool-use routing", "Decide whether to answer directly, call a tool, request information, or decline because the task cannot be answered.", "User request and available context", "Choice(4): direct, tool_call, request_for_info, cannot_answer", "3,652 evaluation requests in the Decision Index projection", "Four actions", "Use the pinned test MCQ file only; do not run synthetic-training generation.", "accuracy, macro-F1, NLL, Brier, ECE"),
    "bfcl": ("Berkeley Function Calling Leaderboard (BFCL)", "System-One runtime tool routing", "Reduce selected BFCL examples to tool selection over runtime-provided candidate descriptions; argument generation is out of scope for Phase 1.", "User request and available tool definitions", "Choice(K), including NONE/NO_TOOL for irrelevance", "Selected public evaluation subset; exact count is fixed by the pinned projection manifest", "Dynamic tool set", "Pin one BFCL version and selected categories; exclude historical versions.", "accuracy and macro-F1 by category; report coverage for NO_TOOL"),
    "banking77_bolt": ("BOLT BANKING77 protocol", "Open-set intent classification overlay", "Apply BOLT known/unknown class partitions and OW-0/OW-1/OW-2 protocols to the canonical BANKING77 raw data.", "Reference to BANKING77 utterances", "Closed known classes; UNKNOWN; or runtime-supplied unseen class", "BOLT processed projection: 13,072", "77 intents", "Primary known75 fold; raw_ref points to system_one/01_banking77/raw.", "known accuracy, unknown AUROC/AUPR, FPR@95TPR, OSCR, supplied-unseen accuracy"),
    "clinc150_bolt": ("BOLT CLINC150 protocol", "Open-set intent classification overlay", "Apply BOLT class holdouts to CLINC intents; this is distinct from native CLINC OOS examples.", "Reference to CLINC utterances", "Closed known classes; UNKNOWN; or runtime-supplied unseen class", "BOLT processed projection: 22,495", "150 intents", "Primary known75 fold; raw_ref points to system_one/02_clinc150_oos/raw.", "known accuracy, unknown AUROC/AUPR, FPR@95TPR, OSCR, supplied-unseen accuracy"),
    "hwu64_bolt": ("BOLT HWU64", "Open-set multi-domain intent classification", "Test whether open-world behavior transfers to a different 64-intent ontology and domain mix.", "User utterance", "Choice(K), UNKNOWN, or runtime-supplied unseen class", "BOLT processed projection: 9,677", "64 intents", "Primary known75 fold; retain BOLT class and example split manifests.", "known accuracy, unknown AUROC/AUPR, FPR@95TPR, OSCR, supplied-unseen accuracy"),
    "stackoverflow_bolt": ("BOLT StackOverflow", "Open-set technical category classification", "Test unseen technical categories to reduce dependence on assistant-intent domains.", "StackOverflow text", "Choice(K), UNKNOWN, or runtime-supplied unseen class", "BOLT processed projection: 19,985", "20 categories", "Primary known75 fold; retain BOLT class and example split manifests.", "known accuracy, unknown AUROC/AUPR, FPR@95TPR, OSCR, supplied-unseen accuracy"),
    "cifar10": ("CIFAR-10", "VLM image classification", "Sanity-check image-to-class decisions at 10-way cardinality.", "Image", "Choice(10) using class names/descriptions", "60,000 images: 50,000 train and 10,000 test", "10 classes", "Use official train/test split; test examples are evaluation-only.", "top-1 accuracy, macro-F1, NLL, Brier, ECE"),
    "oxford_iiit_pets": ("Oxford-IIIT Pets", "VLM fine-grained image classification", "Measure discrimination among 37 cat and dog breeds.", "Image", "Choice(37) using breed names/descriptions", "7,349 images with official train/test split", "37 breeds", "Preserve official split and label mapping; test images are evaluation-only.", "top-1 accuracy, macro-F1, NLL, Brier, ECE"),
    "flowers102": ("Oxford Flowers-102", "VLM high-cardinality fine-grained image classification", "Measure semantic visual classification at 102-way cardinality.", "Image", "Choice(102) using class names/descriptions", "8,189 images with official train/validation/test split", "102 classes", "Preserve official split; test images are evaluation-only.", "top-1 accuracy, macro-F1, NLL, Brier, ECE"),
}

TEMPLATE = """# Dataset: {name}

## 1. Role in this project
{role}.

## 2. Research question
{question}

## 3. Source
Official URL/repository: {source}
Pinned commit/revision: see `SOURCE.lock.yaml` (must be immutable before download)
Downloaded date: not downloaded

## 4. Public availability
Public: yes; access and use remain subject to upstream terms.
License/terms: verify and record upstream license before redistribution.
Redistribution allowed: not assumed; raw artifacts remain local and are not redistributed by this repository.

## 5. Dataset size
Expected: {size}
Observed: not downloaded

## 6. Classes / labels
Expected: {labels}
Class names location: source-defined labels and `processed/` manifest after preparation.

## 7. Original task
See the upstream source and its documentation.

## 8. Our System-One mapping
Input state: {input}
Question: select or score the correct target for the input.
Criteria/options: {mapping}
Expected output: {mapping}

## 9. Example
Canonical examples are recorded in `processed/` only after the source-specific prepare step; no fabricated sample is presented here.

## 10. Benchmark protocol
{protocol}

## 11. Metrics
Primary: {metrics}
Secondary: report per-class results and calibration metrics when target probabilities are available.

## 12. Preprocessing
Source-specific transformations are versioned in `scripts/prepare.py`; raw source files are immutable.

## 13. Leakage rules
Evaluation splits must not enter model training or model selection. Any intentional in-domain training must be called out in the experiment manifest. {leakage}

## 14. Files
`raw/` stores the one canonical raw copy when this dataset owns raw data. `processed/` stores normalized records. `splits/` stores frozen IDs and protocol manifests.

## 15. Checksums
`checksums.sha256` is generated after download; currently no raw files are present.

## 16. Known limitations
Expected counts and protocols are planning values until verified against the pinned source. Label ambiguity, domain bias, and source-specific limitations must be recorded after inspection.

## 17. Reproduction command
`python benchmarks/scripts/download_core.py --dataset {key}` followed by `python benchmarks/{path}/scripts/prepare.py` and `python benchmarks/{path}/scripts/validate.py`.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace generated scaffold files")
    args = parser.parse_args()
    registry = yaml.safe_load((ROOT / "registry.yaml").read_text())
    for source_dir in (ROOT / "sources" / "jevbench", ROOT / "sources" / "bolt", ROOT / "sources" / "bfcl", ROOT / "sources" / "when2call"):
        source_dir.mkdir(parents=True, exist_ok=True)
    for key, item in registry["datasets"].items():
        path = ROOT / item["path"]
        if key.endswith("_bolt"):
            (path / "splits").mkdir(parents=True, exist_ok=True)
            (path / "processed").mkdir(parents=True, exist_ok=True)
            write(path / "PROTOCOL.md", protocol_doc(key), args.force)
        if key in {"banking77_bolt", "clinc150_bolt"}:
            (path / "raw_ref.yaml").parent.mkdir(parents=True, exist_ok=True)
            raw = item["raw_ref"]
            ref = f"dataset: {raw.split('/')[-2]}\nraw_path: ../../{raw}\n"
            write(path / "raw_ref.yaml", ref, args.force)
        else:
            for part in ("raw", "processed", "splits"):
                (path / part).mkdir(parents=True, exist_ok=True)
        (path / "scripts").mkdir(parents=True, exist_ok=True)
        name, role, question, input_state, mapping, size, labels, protocol, metrics = DETAILS[key]
        source = item["source"]
        doc = TEMPLATE.format(name=name, role=role, question=question, input=input_state, mapping=mapping,
                              size=size, labels=labels, protocol=protocol, metrics=metrics,
                              leakage="" if key not in {"banking77_bolt", "clinc150_bolt"} else "Unseen class labels must never appear in training class lists.",
                              source=source, key=key, path=item["path"])
        write(path / "DATASET.md", doc, args.force)
        lock = lock_doc(key, item)
        write(path / "SOURCE.lock.yaml", lock, args.force)
        write(path / "checksums.sha256", "# Generated by checksum_all.py after raw files are downloaded.\n", args.force)
        status = {"downloaded": False, "validated": False, "prepared": False, "examples": None,
                  "classes": item.get("classes"), "source_revision": None, "checksum_verified": False}
        write(path / "status.json", json.dumps(status, indent=2) + "\n", args.force)
        for action in ("download", "prepare", "validate"):
            wrapper = f'''#!/usr/bin/env python3\n"""Per-dataset {action} entry point."""\nfrom pathlib import Path\nimport runpy\nimport sys\naction_script = Path(__file__).resolve().parents[3] / "scripts" / "dataset_action.py"\nsys.argv = [str(action_script), "--dataset", "{key}", "--action", "{action}", *sys.argv[1:]]\nrunpy.run_path(str(action_script), run_name="__main__")\n'''
            write(path / "scripts" / f"{action}.py", wrapper, args.force)
    print(f"Scaffolded {len(registry['datasets'])} benchmark definitions under {ROOT}.")


def write(path: Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def lock_doc(key: str, item: dict) -> str:
    return f'''dataset_id: {key}\nsource_type: repository_or_hub\nsource:\n  url: "{item['source']}"\n  revision: PIN_REQUIRED\nlicense: VERIFY_UPSTREAM_TERMS\nlicense_url: VERIFY_UPSTREAM_TERMS\ndownloaded_at: null\nexpected:\n  examples: {item.get('expected_examples', 'null')}\n  classes: {item.get('classes', 'null')}\nraw_hashes: {{}}\ntransform_version: v1\nstatus: planned\nnotes:\n  raw_unique: {str(item['raw_unique']).lower()}\n'''


def protocol_doc(key: str) -> str:
    return f"""# {DETAILS[key][0]} protocol\n\nThis directory contains protocol metadata only; it owns no raw-data copy.\n\nMaterialize `known75` first. Define three derived sets:\n\n- **OW-0 Closed:** candidates are known classes only.\n- **OW-1 Unknown absent:** known classes plus `UNKNOWN`; target is `UNKNOWN` for held-out classes.\n- **OW-2 Runtime supplied:** known classes, the unseen class description, and `UNKNOWN`; target is the supplied unseen class.\n\nThe primary split must keep known and unseen class sets disjoint. Record exact BOLT revision, fold, example IDs, and class-description provenance in `splits/`. Never include unseen labels in training class lists.\n"""


if __name__ == "__main__":
    main()
