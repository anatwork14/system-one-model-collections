#!/usr/bin/env python3
"""Generate consistent method cards, artifact locks, and starter scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METHODS = [
    dict(path="released/00_semif", name="SemIf / OpenJev", status="Released implementation; frozen base-model baseline", family="A — direct option-token scoring", base="Qwen/Qwen3.5-4B", rev="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", params="4B class; multimodal checkpoint", mode="Frozen direct typed-option logits. It reproduces the semantic-decision interface and does not claim to reproduce Jev's undisclosed architecture.", arch="state + question + typed options → prompt → one model forward → allowed option logits → softmax", inp='JSONL decision with state, question, and options/criteria.', out="Probability per typed option; serialized response includes scores and timing.", ar="No answer-token generation in direct mode.", prob="Next-token vocabulary logits gathered at the decision position for declared option tokens, then normalized over the candidate set.", train="None.", trainable="None (0 trainable parameters).", frozen="Entire Qwen3.5-4B checkpoint.", loss="None.", data="None.", config="No training configuration.", artifacts="Base tokenizer/model only, shared via symlink. Source examples/fixtures are under upstream.", infer="See scripts/smoke.sh; CLI: semif-score --mode direct --model ../../shared/models/qwen3.5-4b --input upstream/examples/decisions.jsonl --output results/smoke.jsonl (adjust relative path from method root as shown by the script).", resources="4B checkpoint plus runtime/KV buffers; exact VRAM and latency are unmeasured here and must be recorded on the target host.", limits="Depends on prompt and tokenizer option-token mapping; softmax is not calibration; no generated reasoning; candidate count/context constraints follow upstream implementation.", hypothesis="Can frozen pretrained option logits provide a useful zero-training System-One baseline?", repo="https://github.com/iwillcodeu/openjev", branch="master", checkpoint="none", dataid="none"),
    dict(path="released/01_simple_jev", name="Simple Jev", status="Released implementation; zero-shot inference by default", family="A — direct option-token scoring / classifier serving", base="Qwen/Qwen3.5-4B", rev="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", params="4B class; multimodal checkpoint", mode="Prompt compiler and classifier scorer expose typed Choice, Score, and Boolean decisions over compatible Hugging Face models. Server constructs structured output from next-token logits; model does not generate JSON.", arch="typed request → prompt compiler → HF model next-token logits → response scorer → typed probabilities", inp="Classifier API request containing state and named typed questions with instructions and candidate values.", out="Structured typed scores/probabilities; server emits JSON response.", ar="No structured answer generation; reads next-token logits.", prob="Scorer maps allowed answer/option token logits into the requested typed distribution.", train="None for default serving. RFDT training workflow is available upstream and is separate from baseline inference.", trainable="None at inference.", frozen="Selected HF model at inference.", loss="None for default inference; optional RFDT workflow has its own upstream objective.", data="None required for default serving.", config="Server/runtime configuration only; use pinned model ID and the upstream Qwen dense 4B prompt policy if supported by checked-out revision.", artifacts="Shared base model. No method-specific checkpoint for baseline.", infer="See scripts/smoke.sh and upstream README; install the upstream HF server package and start with --model ../../shared/models/qwen3.5-4b.", resources="4B checkpoint plus server buffers; measure on target hardware. Input context and batch limits are server-configured.", limits="Compatibility depends on prompt policy, tokenizer, and allowed answer-token definitions; probabilities are not automatically calibrated. API behavior may change across upstream revisions.", hypothesis="Does a reusable typed-classifier serving API make direct scoring practical beyond a research CLI?", repo="https://github.com/featherless-ai/simple-jev", branch="main", checkpoint="none", dataid="none"),
    dict(path="released/02_tev1_4b", name="Tev1-4B", status="Published checkpoint", family="B — autoregressive answer-token SFT", base="Qwen/Qwen3.5-4B", rev="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", params="4B class; post-trained multimodal checkpoint", mode="LoRA supervised fine-tuning specializes the model to choose a letter for a structured decision. The standard LM head is retained.", arch="state + question + labeled options → Qwen backbone + LoRA → next answer letter → application maps letter to option", inp="System instruction and structured state, question, and 2–24 labeled options.", out="One option letter (then application maps it to the semantic key).", ar="Yes, autoregressive; intended output is one letter, not a probability vector.", prob="If available, token log-probabilities can be exposed for letter choices; ordinary generated answer itself is the published interface. Do not assume calibrated option probabilities.", train="LoRA SFT, one epoch in published starting recipe.", trainable="LoRA adapters; existing LM head is used and not described as a new head.", frozen="Base weights outside LoRA.", loss="Full-vocabulary next-token CE over supervised answer output.", data="Published new-v1: 37,840 train and 4,568 development examples. Dataset includes public-source tasks and synthetic authored decisions. Exact sources/build manifests are upstream; do not label the full set human-reviewed.", config="Published preview recipe: rank 8, alpha 16, LR 5e-5, batch 8, gradient accumulation 1, sequence length 2048, 1 epoch, packing, seed 42; target training environment is Together H100. Recipe may not match a historical run byte-for-byte.", artifacts="Full model checkpoint from togethercomputer/Tev1-4B-experimental; tokenizer/config included in HF repo; base source repo and recipe under upstream.", infer="See scripts/smoke.sh; use Transformers generation or the upstream example client. Local generation command is prepared in scripts/infer.py.", resources="Published checkpoint is about 9.34 GB on Hub; runtime VRAM/latency depend on dtype, context, and serving engine; measure locally.", limits="Autoregressive output adds decoding; one-letter output has no inherent typed calibration; benchmark is not directly comparable to non-generative probability readout without protocol alignment.", hypothesis="Does decision-focused LoRA SFT improve answer-letter selection over frozen direct logits?", repo="https://github.com/togethercomputer/tev1", branch="main", checkpoint="togethercomputer/Tev1-4B-experimental", dataid="new-v1"),
    dict(path="released/03_jevk5_4b", name="JevK5-4B", status="Published merged checkpoint", family="C — distilled backbone with option-logit readout", base="Qwen/Qwen3.5-4B", rev="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", params="4B class", mode="Distilled LoRA is merged into Qwen3.5-4B; runtime uses a SemIf-style softmax over answer-letter logits with a calibration temperature.", arch="state + typed question + candidates → merged Qwen → option-letter logits → temperature-scaled softmax", inp="Typed state/question/options, encoded using the project's prompt format.", out="Probability for every candidate; no free-form answer required at runtime.", ar="No generated answer sequence at runtime; single forward option-logit readout (multiple passes may be used for oversized option sets).", prob="Softmax of allowed answer-letter next-token logits after published temperature calibration.", train="LoRA decision distillation; teacher-generated decisions plus public replay. Current v0.3 reports 17,408 teacher questions and 30,052 public replay items.", trainable="LoRA parameters during training; merged into released backbone.", frozen="Base parameters except LoRA during training; entire merged checkpoint is fixed at inference.", loss="Option-letter cross-entropy/distillation as described by the tagged training recipe; v0.1/v0.2 settings are not asserted to describe current v0.3 exactly.", data="Teacher generated (Qwen3.6-27B and GPT-6 Luna per current project description) plus 26 public train splits. Generated local artifacts must be distinguished from upstream public source data and released checkpoint.", config="Version-specific. v0.1/v0.2 recipe reports rank 16, LR 3e-5, two epochs; consult exact pinned upstream tag/config before reproducing current v0.3.", artifacts="Merged HF checkpoint alibiserikbay/JevK5 and tokenizer/config; repo provides training/inference implementation.", infer="See scripts/smoke.sh and upstream README; call the package/server or direct scoring helper with shared base tokenizer assets as documented upstream.", resources="4B checkpoint plus inference runtime; exact VRAM/latency should be measured on target hardware.", limits="English-focused, option/context limits and temperature are version-specific; softmax calibration may shift with new domains and option counts; teacher labels may inherit model errors.", hypothesis="Does distillation plus calibrated option-logit readout improve typed decisions while retaining one-pass inference?", repo="https://github.com/allebee/jevk5", branch="main", checkpoint="alibiserikbay/JevK5", dataid="v0.3 teacher + public replay"),
    dict(path="released/04_kev_4b", name="Kev-4B", status="Published adapter and pointer head", family="D — candidate pointer scoring", base="Qwen/Qwen3.5-4B-Base", rev="1001bb4d826a52d1f399e183466143f4da7b741b", params="4B class; model card reports 4.7B total including vision tower", mode="Qwen hidden states for decision marker and candidates feed a custom pointer head. Final decisions come from candidate scores, not LM vocabulary logits.", arch="shared state + question/decision marker + candidate spans → Qwen + LoRA hidden states → pointer head → candidate softmax", inp="State with one or more typed questions and candidate descriptions serialized using Kev's delimiter format.", out="Probability distribution over candidates/typed answers.", ar="No answer text generation; one forward decision pass.", prob="Softmax of pointer-head candidate scores.", train="LoRA plus pointer-head supervised training; published recipe uses decision-v7 and two epochs; later release also includes a delta update.", trainable="LoRA rank 16 (reported 33.8M parameters) and pointer head.", frozen="Base parameters outside LoRA during published PEFT training.", loss="Pointer/candidate cross-entropy.", data="decision-v7 public benchmark/task mixtures plus generated policy/rule data; later update reports 11,320 new records plus 4,000 replay examples. Not all later generated labels are public human labels.", config="Recipe: LoRA r=16 over attention, MLP, and DeltaNet projections; LR 5e-5, two epochs; later delta is separately documented upstream. Exact checkpoint revision is in the artifact lock.", artifacts="HF adapter and pointer head, plus Qwen3.5-4B-Base shared base. Model card files and tokenizer/config metadata must be captured by fetch manifest.", infer="See scripts/smoke.sh and upstream model card/server entry point; adapter and head load on weights/base_model.", resources="Base model + adapter + custom head. VRAM and latency depend on dtype/context/implementation; do not copy a number from another host.", limits="Requires custom architecture/runtime and exact row/mask serialization; candidate descriptions and delimiter/token IDs matter; generated data and calibration can limit OOD behavior.", hypothesis="Does a learned pointer readout with modest backbone adaptation outperform vocabulary-token scoring?", repo="https://github.com/jaredpalmer/kev", branch="main", checkpoint="jaredpalmer/kev-4b", dataid="decision-v7 + later delta"),
    dict(path="retrain/10_nimble_4b", name="Nimble-4B reproduction", status="Proposed reproduction; not trained", family="B — candidate-only LM-logit CE", base="Qwen/Qwen3.5-4B", rev="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", params="4B class", mode="Adapt the published schema-aware candidate-logit training recipe from Qwen3.5-9B to 4B.", arch="schema decision → Qwen + LoRA → allowed candidate logits only → candidate softmax", inp="Schema-aware record with state/context, field/question, and allowed choices.", out="Candidate probability distribution.", ar="No generated output at inference; candidate scoring from one forward pass.", prob="Softmax over the allowed candidate-token logits.", train="LoRA with candidate-only cross-entropy; one epoch.", trainable="LoRA r=16; standard LM head is retained.", frozen="Qwen weights outside LoRA.", loss="L = -log(exp(z_y) / sum_{c in C} exp(z_c)); disallowed vocabulary tokens do not compete in the decision loss.", data="2,676 train and 324 frozen evaluation examples. Ten domains and Choice/Noul/Score tasks. Labels are synthetic/model-checked and not human-reviewed. Preserve upstream license and provenance.", config="LR 5e-5; seed 17; batch 2; accumulation 4 (effective batch 8); max length 2048; one epoch; BF16; LoRA rank 16. Backbone size is the intended change from published 9B recipe.", artifacts="No published 4B checkpoint. Upstream Nimble trainer/data; local adapter output in weights/nimble-4b.", infer="After training: bash scripts/eval.sh. Train entry point: bash scripts/train.sh. No training is run by workspace setup.", resources="4B base plus activations/optimizer states; expected VRAM depends on checkpointing, precision, and GPU; measure on target host. Disk includes base and adapter.", limits="Small synthetic/model-checked dataset; domain coverage and label quality constrain conclusions; frozen holdout must never tune hyperparameters; tokenizer export from 9B may need regeneration for 4B.", hypothesis="Does candidate-only decision CE provide an advantage over ordinary answer-token SFT on the same examples?", repo="https://github.com/bespokelabsai/nimble", branch="original-2676", checkpoint="none; official checkpoint is Qwen3.5-9B based", dataid="2,676 train + 324 held-out"),
    dict(path="retrain/11_clm_style_4b", name="CLM-style-4B reproduction", status="Proposed controlled reproduction; not trained", family="E — contrastive state/action representations", base="Qwen/Qwen3.5-4B", rev="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", params="4B class", mode="Freeze Qwen and train state/action projection heads so compatible action descriptions can be scored by similarity.", arch="state → frozen Qwen → state projection z_s; candidate action → frozen Qwen → action projection z_a; dot products → softmax", inp="One state and candidate action/class descriptions.", out="Similarity and normalized candidate probability.", ar="No autoregressive answer generation; encoder-style forward passes for state and actions.", prob="Softmax of state/action similarity scores divided by learned or fixed temperature.", train="InfoNCE contrastive training on controlled Nimble-derived pairs.", trainable="State and action projections; optional scalar temperature.", frozen="Entire Qwen3.5-4B backbone.", loss="Batch InfoNCE: -log exp(sim(z_s,z_pos)/τ) divided by the sum over positive and negative action candidates.", data="Derive positive state→correct-candidate pairs from the same 2,676 Nimble train examples; other allowed candidates become hard negatives. Do not use the 324 holdout for training. Original labels are synthetic/model-checked.", config="Proposed: seed 17; same train IDs and effective batch as controlled arms; projection optimizer/LR to be frozen in configs/clm.yaml before first run; 2048-token input cap. This is a reproduction design, not official CLM training config.", artifacts="No Qwen3.5-4B published checkpoint. CLM upstream implementation is a reference; local projection heads will be saved under weights/state_head and weights/action_head.", infer="After training: python scripts/infer.py --config configs/clm.yaml. Training launcher is scripts/train.sh; not executed here.", resources="Frozen 4B encoder dominates VRAM; two tower inputs may be batched/cached. Exact memory/latency unmeasured.", limits="A frozen pretrained base may not produce aligned state/action vectors without enough contrastive data; pair construction and negatives strongly affect results; 2,676 examples are small.", hypothesis="Can a frozen encoder with lightweight contrastive heads support dynamic candidate descriptions and novel classes?", repo="https://github.com/Contrastive-LM/CLM", branch="main", checkpoint="none", dataid="controlled Nimble-derived pairs"),
    dict(path="retrain/12_pointer_head_only_4b", name="Pointer-head-only-4B", status="Proposed ablation; not trained", family="D — frozen-backbone pointer scoring", base="Qwen/Qwen3.5-4B-Base", rev="1001bb4d826a52d1f399e183466143f4da7b741b", params="4B class", mode="Reuse Kev candidate representation/readout with a fully frozen Qwen base and train only the pointer head.", arch="state + candidates → frozen Qwen hidden states → trainable pointer head → candidate softmax", inp="Kev-format state/question/candidate rows, converted from controlled Nimble decisions.", out="Candidate probability distribution.", ar="No generated answer sequence.", prob="Pointer-head score softmax across candidates.", train="Pointer CE; head-only ablation.", trainable="Pointer head only.", frozen="All Qwen3.5-4B-Base backbone weights.", loss="Candidate cross-entropy over pointer scores.", data="Initial controlled set is the 2,676 Nimble train records, converted to Kev's row format; optional Kev decision-v7 extension is a separate experiment and must not be mixed into the first controlled result.", config="Same train IDs/split/seed as other controlled arms; proposed seed 17. Head optimizer settings are in configs/pointer_head.yaml and must be frozen before full run.", artifacts="Kev upstream pointer implementation as read-only source; local trained head under weights/head.", infer="After training: python scripts/infer.py --config configs/pointer_head.yaml. No training executed.", resources="Frozen 4B base plus activations and small head; exact VRAM/latency unmeasured.", limits="Head-only result depends on representation quality and exact Kev serialization/masks; comparison with Kev LoRA must share initialization, data, and evaluation.", hypothesis="Is a custom decision head sufficient, or is backbone adaptation needed?", repo="https://github.com/jaredpalmer/kev", branch="main", checkpoint="none", dataid="controlled Nimble-derived decisions"),
    dict(path="retrain/13_kev_fullft_4b", name="Kev-style Full-FT-4B", status="Proposed upper-bound reproduction; not trained", family="D — pointer scoring with full backbone fine-tuning", base="Qwen/Qwen3.5-4B-Base", rev="1001bb4d826a52d1f399e183466143f4da7b741b", params="4B class", mode="Keep Kev pointer architecture and train every Qwen backbone parameter plus the pointer head.", arch="state + candidates → fully trainable Qwen → pointer head → candidate softmax", inp="Kev-format state/question/candidate rows.", out="Candidate probability distribution.", ar="No answer-text generation.", prob="Pointer-head score softmax.", train="Full fine-tuning with pointer candidate CE.", trainable="All backbone weights and pointer head.", frozen="None, aside from any explicitly frozen vision modules in the exact text path; default recipe trains the whole text backbone.", loss="Candidate pointer cross-entropy.", data="Same 2,676 Nimble train rows and frozen 324 evaluation rows for the first controlled comparison; labels synthetic/model-checked.", config="Controlled seed 17 and same split. FSDP/DeepSpeed, activation checkpointing, microbatch, and optimizer settings must be set in configs/full_ft.yaml before run. Do not launch before smaller arms pass their data/model wiring checks.", artifacts="Kev upstream pointer implementation; local full checkpoint under weights/fullft.", infer="After training: python scripts/infer.py --config configs/full_ft.yaml. No training executed.", resources="Highest memory and compute arm; distributed sharding/offload may be required. Hardware sizing is deferred to the target environment.", limits="Expensive and may overfit the small synthetic set; full fine-tuning comparison is only meaningful with identical initialization/data/splits and independent checkpoint evaluation.", hypothesis="What quality ceiling does full backbone adaptation add over LoRA or head-only pointer training?", repo="https://github.com/jaredpalmer/kev", branch="main", checkpoint="none", dataid="controlled Nimble-derived decisions"),
    dict(path="retrain/14_vlm_systemone_4b", name="VLM System-One 4B", status="NOT_STARTED — placeholder only", family="F — multimodal decision model", base="Qwen/Qwen3.5-4B multimodal checkpoint (exact selected revision to be pinned later)", rev="deferred", params="4B class plus bundled vision components", mode="Port the best validated text decision mechanism to image-conditioned decisions after text experiments.", arch="image + text state + typed candidates → multimodal Qwen → selected decision readout", inp="Image(s), textual state/question, and typed candidate choices.", out="Typed decision probabilities.", ar="To be selected with winning text mechanism.", prob="To be selected and explicitly calibrated/evaluated.", train="Not started. Planned V0 frozen VLM/direct logits; V1 frozen VLM/head; V2 language LoRA/head; V3 connector + language LoRA/head; V4 upper vision blocks; V5 full VLM fine-tuning.", trainable="None selected yet.", frozen="None selected yet.", loss="To be selected after text phase.", data="No dataset selected. Keep image train/validation/test splits and image provenance/license explicit.", config="Not started.", artifacts="None.", infer="Not available.", resources="Not estimated.", limits="Deferred; image-text leakage, visual preprocessing, and multimodal calibration need separate protocol.", hypothesis="Can the best text decision interface extend to image-grounded System-One decisions?", repo="none", branch="none", checkpoint="none", dataid="none"),
]

def method_md(m):
    return f'''# {m['name']}

## 1. Status

{m['status']}

## 2. Family

{m['family']}

## 3. Base model

Model: `{m['base']}`  
Revision: `{m['rev']}`  
Parameter count: {m['params']}  
Base vs instruct: {"Base" if "-Base" in m['base'] else "post-trained/instruct family checkpoint; verify exact model card"}

## 4. Core idea

{m['mode']}

## 5. Architecture

```text
{m['arch']}
```

## 6. Input

{m['inp']}

## 7. Output

{m['out']}

## 8. Autoregressive?

{m['ar']}

## 9. Decision probability

{m['prob']}

## 10. Training

{m['train']}

## 11. Trainable modules

{m['trainable']}

## 12. Frozen modules

{m['frozen']}

## 13. Training objective

{m['loss']}

## 14. Training data

{m['data']}

## 15. Training configuration

{m['config']}

## 16. Published artifacts

{m['artifacts']}

## 17. Local artifacts

- `upstream/`: pinned source repository, treated as read-only.
- `weights/`: method checkpoint/adapter/head, or symlink to `shared/models/`.
- `data/`: raw/processed/splits/manifests; source data is not edited in place.
- `configs/`, `scripts/`, `results/`: local reproducibility assets and outputs.
- Artifact download status and exact revisions: `ARTIFACTS.lock`.

## 18. Inference

{m['infer']}

## 19. Expected resources

{m['resources']}

## 20. Known limitations

{m['limits']}

## 21. Why it matters scientifically

{m['hypothesis']}

## 22. Source provenance

Repository: [{m['repo']}]({m['repo']})  
Requested ref: `{m['branch']}`  
Resolved source commit: recorded by `tools/fetch_all.py` after network fetch; see `ARTIFACTS.lock`.  
Checkpoint: `{m['checkpoint']}`  
Checkpoint Hub revision: resolved to a full immutable commit SHA during fetch where applicable.  
Data recipe: `{m['dataid']}`  
Downloaded at: recorded after successful fetch.  
Local changes: workspace metadata, wrappers, and configs only; upstream files are not modified.
'''

def lock_yaml(m):
    return f'''format: 1
method: {m['name'].lower().replace(' ', '_').replace('/', '_')}
status: {m['status']}
source:
  repo: {m['repo']}
  ref: {m['branch']}
  commit: null  # tools/fetch_all.py resolves and writes the immutable SHA
base:
  model: {m['base']}
  revision: {m['rev']}
checkpoint:
  model: {m['checkpoint']}
  revision: null  # resolved from Hub model_info during fetch; none for baseline-only methods
data:
  recipe: {m['dataid']}
  manifest: data/manifests/manifest.json
  hashes: not-downloaded
downloaded_at: null
local_changes:
  - local method documentation/configs/scripts only
  - upstream source remains unmodified
'''

for m in METHODS:
    path = ROOT / m['path']
    for d in ('upstream', 'weights', 'data/raw', 'data/processed', 'data/splits', 'data/manifests', 'configs', 'scripts', 'results'):
        (path / d).mkdir(parents=True, exist_ok=True)
    (path / 'METHOD.md').write_text(method_md(m), encoding='utf-8')
    (path / 'ARTIFACTS.lock').write_text(lock_yaml(m), encoding='utf-8')
    (path / 'data/DATASET.md').write_text(f'''# Data inventory — {m['name']}

## Source

Recipe: `{m['dataid']}`. Source repo/artifact and resolved revision are tracked in `../ARTIFACTS.lock`.

## Local organization

- `raw/`: byte-preserving copies of source files.
- `processed/`: derived examples with transformation script/version.
- `splits/`: frozen row/family IDs and split purpose.
- `manifests/manifest.json`: counts, licenses, label origin, hashes, and model-selection use.

No local data files have been fetched yet unless a file is explicitly listed in the manifest. Never use frozen evaluation records for training or model selection.
''', encoding='utf-8')
    (path / 'data/manifests/manifest.json').write_text('''{\n  "status": "not_downloaded",\n  "files": [],\n  "notes": "Populate hashes/counts/license/provenance after verified fetch."\n}\n''', encoding='utf-8')
    (path / 'scripts/smoke.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
echo "Smoke inference is prepared by METHOD.md but requires downloaded artifacts and a supported runtime." >&2
echo "Method: $(basename "$(dirname "$0")")" >&2
exit 2
'''.replace('+',''), encoding='utf-8')
    (path / 'scripts/smoke.sh').chmod(0o755)
    (path / 'upstream/UPSTREAM_README.txt').write_text(f'''Source checkout target: {m['repo']} @ {m['branch']}
Checkout and resolved commit are recorded in ../ARTIFACTS.lock.
Do not edit files in this directory.
'''.replace('+',''), encoding='utf-8')

for rel, txt in {
    'retrain/10_nimble_4b/REPRODUCTION.md': '''# Reproduction protocol — Nimble-4B

This is a Qwen3.5-4B adaptation of Nimble's public Qwen3.5-9B candidate-logit trainer. Keep the 324-row evaluation set frozen. Run preparation first, inspect `data/manifests/manifest.json`, then perform a 32–64 example overfit sanity run before the full epoch. That sanity run is future work and is not part of workspace preparation.

The only planned method change is backbone size. Because candidate token IDs and any cached token exports depend on tokenizer fingerprint, regenerate/verify token exports using the pinned 4B tokenizer before training.
''',
    'retrain/11_clm_style_4b/REPRODUCTION.md': '''# Controlled CLM-style reproduction

This is a small controlled adaptation, not a reproduction of CLM's 91M-scale training. Use only the Nimble train split, convert the correct candidate to a positive state/action pair, and treat remaining candidates as negatives. Freeze the base encoder and fit projections/temperature. Record pair-generation code and hashes.
''',
    'retrain/12_pointer_head_only_4b/REPRODUCTION.md': '''# Pointer-head-only ablation

Keep Kev's pointer serialization and head implementation as the reference. Freeze every backbone tensor. Train only the pointer head on the shared Nimble-derived train rows. Compare with LoRA+pointer using identical initialization, rows, seed, and evaluation protocol.
''',
    'retrain/13_kev_fullft_4b/REPRODUCTION.md': '''# Kev-style full fine-tuning upper bound

This arm uses the same candidate serialization and pointer objective as the PEFT arm but updates every backbone parameter. Keep it last in the run order. Freeze configuration, data IDs, and evaluation before launching; record any sharding/offload behavior in the result metadata.
''',
    'retrain/14_vlm_systemone_4b/STATUS.md': '''# Status: NOT_STARTED

Placeholder only. Select the winning text mechanism and a licensed, leakage-controlled image decision dataset before choosing an exact multimodal checkpoint revision or training configuration.
'''
}.items():
    (ROOT / rel).write_text(txt, encoding='utf-8')

