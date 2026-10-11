
# VLM System-One 4B — Complete Training Methodology

## 1. Objective

The goal is to convert a pretrained multimodal Qwen3.5-4B model into a **non-generative multimodal decision model** that receives:

\[
x=(I_{1:m},S,Q,C)
\]

where:

- \(I_{1:m}\): one or more images;
- \(S\): textual or structured state;
- \(Q\): a focused decision question;
- \(C=\{c_1,\ldots,c_K\}\): typed candidate choices;

and returns:

\[
P(Y=c_i\mid I,S,Q,C)
\]

without autoregressively generating an answer.

The model therefore uses an autoregressive Qwen backbone internally, but **decision inference itself is non-autoregressive**:

\[
\text{multimodal prefill}
\rightarrow
\text{decision score}
\rightarrow
\text{probabilities}.
\]

No answer string is decoded.

---

# 2. Base model

## Primary checkpoint

Use:

```text
Qwen/Qwen3.5-4B
```

rather than leaving the checkpoint unspecified.

Qwen3.5 is natively multimodal, trained over interleaved text, image and video. The 4B checkpoint uses a 32-layer, 2560-hidden-dimension language backbone together with a vision encoder; the Hub artifact is approximately 5B total parameters once the vision components are included.

Its vision configuration currently contains:

```text
vision depth        = 24
vision hidden size  = 1024
vision output       = 2560
patch size          = 16
spatial merge       = 2
```

which makes its visual representation directly compatible with the 2560-dimensional language state.

The vision tower includes a patch-merger projection ending in the language hidden dimension, making that merger a natural low-cost adaptation point for V3.

## Base-model ablation

Also evaluate:

```text
Qwen/Qwen3.5-4B-Base
```

but do not make it the default.

The primary model should be the post-trained checkpoint because this project is trying to learn a decision surface, not re-learn instruction semantics and multimodal understanding.

The Base checkpoint is scientifically useful as an ablation:

\[
\text{Base vs Post-trained}
\]

to determine whether Qwen's post-training improves or harms decision specialization/calibration.

## Revision

Never train against `main`.

At download time:

```python
model_info("Qwen/Qwen3.5-4B").sha
```

should resolve the current Hub revision to a complete immutable SHA and write it into:

```text
ARTIFACTS.lock
```

alongside processor, tokenizer, Transformers version and dataset revisions.

---

# 3. Core architecture

## 3.1 Recommended architecture: candidate cross-encoder

Do not make the final model generate:

```text
A
B
C
D
```

or JSON.

Instead, transform every candidate into a separate decision branch:

```text
image(s)
    +
shared state
    +
question
    +
one candidate
        ↓
Qwen3.5 multimodal backbone
        ↓
last decision hidden state
        ↓
shared scalar decision head
        ↓
candidate score s_i
```

For \(K\) candidates:

\[
s_i=f_\theta(I,S,Q,c_i).
\]

Then normalize across candidates:

\[
p_i=
\frac{\exp(s_i/T)}
{\sum_{j=1}^{K}\exp(s_j/T)}.
\]

Here \(T\) is learned later on a completely separate calibration set.

### Decision head

Use the simplest head first:

\[
s_i=w^\top h_i+b,
\]

where

\[
h_i\in\mathbb{R}^{2560}
\]

is the final non-padding hidden state.

A linear head should be the default. Do not introduce an MLP until an ablation shows that one is necessary.

This architecture is already supported naturally by Transformers because `Qwen3_5ForSequenceClassification` accepts multimodal inputs including `pixel_values`, `image_grid_thw`, video and text.

Therefore the final model can avoid the enormous 248K-token LM output projection entirely.

That is useful both conceptually and computationally:

```text
Qwen generative VLM
2560 hidden
→ 248,320 vocabulary logits
→ decode
```

becomes:

```text
Qwen System-One
2560 hidden
→ 1 decision logit
→ softmax over supplied candidates
```

---

# 4. Why candidate scoring instead of answer-letter logits

A/B/C/D logits are useful as a baseline, but they should not be your main mechanism.

LLMs have well-documented selection and positional biases in multiple-choice tasks. Changing option order alone can substantially change their prediction.

The candidate-cross-encoder formulation avoids this:

```text
candidate key = metadata only

model sees:
"click the submit button"

model does NOT see:
"Option A: click the submit button"
```

Candidate order therefore does not determine its semantic representation.

For any permutation \(\pi\):

\[
P(c_i\mid C)
=
P(c_i\mid \pi(C))
\]

up to numerical precision, because every candidate is independently scored by the same function.

This is a much stronger System-One interface.

---

# 5. Transfer from the winning text System-One model

This is the most important part of your research methodology.

The VLM experiment must **inherit the best text decision mechanism rather than redesigning it**.

There are three cases.

## Case A — scalar scorer wins text experiments

If the text winner uses:

```text
Qwen → hidden state → scalar score
```

copy:

```text
language decision adapter
decision head
decision input template
```

directly into the multimodal model.

Initialize:

```text
vision tower     ← original Qwen3.5 VLM
vision merger    ← original Qwen3.5 VLM
language model   ← text System-One winner
decision head    ← text System-One winner
```

This is the preferred transfer experiment.

---

## Case B — NLI/entailment scorer wins

Suppose the text model outputs:

```text
contradiction
neutral
entailment
```

for each candidate.

Keep the same three-way head.

For candidate \(c_i\), define:

\[
s_i =
z_i^{entail}
-
\log
\left(
e^{z_i^{neutral}}
+
e^{z_i^{contradiction}}
\right).
\]

Then:

\[
P(c_i)=\text{softmax}(s_1,\ldots,s_K).
\]

Conceptually:

> The image + state entails which candidate?

This is a particularly elegant multimodal System-One formulation.

---

## Case C — native LM-token logits win

Preserve it as the main transfer mechanism if the text evidence genuinely shows it superior.

However, still train and evaluate the candidate-scorer baseline because token/position bias is an important confound.

---

# 6. Input representation

A candidate branch should have one canonical format.

Example:

```text
[IMAGE(S)]

State:
{canonical state}

Decision type:
choice

Question:
Which action should be performed next?

Candidate:
Click the Submit button.

Evaluate whether this candidate is the correct decision for the given state.
```

Do not include:

```text
Candidate A
Candidate B
Candidate 1
Candidate 2
```

unless candidate identifiers themselves are semantically relevant.

For structured state objects, use deterministic serialization:

```json
{
  "current_task": "...",
  "previous_action": "...",
  "status": "..."
}
```

with stable key ordering.

Do not automatically convert the image into captions and feed those captions as text; that would create a language shortcut and defeat the multimodal experiment.

---

# 7. Typed decision interface

Support the same three families across text and multimodal phases.

## Choice

Input:

```text
question
candidate_1
...
candidate_K
```

Return:

```json
{
  "selected": "...",
  "probabilities": {
    "...": 0.73,
    "...": 0.19,
    "...": 0.08
  }
}
```

---

## Noul

Represent yes/no internally as two candidates:

```text
TRUE:
The statement is supported by the state.

FALSE:
The statement is not supported by the state.
```

Then:

\[
P(\text{true}) =
\frac{e^{s_T/T}}
{e^{s_T/T}+e^{s_F/T}}.
\]

This keeps one unified model architecture.

---

## Score

For ordered levels:

```text
low
medium
high
critical
```

score all levels as ordinary candidates.

Return both:

\[
P(Y=l_j)
\]

and expected normalized score:

\[
E[Y]=\sum_j p_jv_j.
\]

An ordinal auxiliary loss can be added later, but ordinary candidate CE should remain the primary baseline.

---

# 8. Explicit abstention

The model should optionally support:

```text
NONE_OF_THE_ABOVE
```

or:

```text
INSUFFICIENT_EVIDENCE
```

as a real candidate.

This is preferable to inventing abstention after inference using a fixed probability threshold alone.

Training examples should include cases where:

- no candidate is correct;
- the required visual information is missing;
- the object is occluded;
- the image belongs to another state;
- resolution is insufficient;
- candidates are all plausible but unsupported.

Do not make these cases trivially identifiable from formatting.

Approximately 10–15% of decision training questions can contain genuine abstention examples.

---

# 9. Training dataset

A useful target is approximately:

\[
100K-300K
\]

typed decision questions.

The number of images can be much smaller because one image/state can create several independent questions.

Current open System-One work gives useful scale references: OneJev reports roughly 99K training questions and trains Qwen3.5 models using CE+Brier with a frozen vision tower; OmniJev reports a substantially larger heterogeneous multimodal decision mixture using LoRA, decision heads and separate calibration.

## Recommended mixture

Approximately:

| Component | Share |
|---|---:|
| Natural multimodal reasoning | 25% |
| Visual perception / spatial decisions | 15% |
| UI / screenshot / action selection | 20% |
| Diagram / chart / document understanding | 10% |
| Visual entailment / binary decisions | 10% |
| Counterfactual / unknown / abstention | 10% |
| Text-System-One replay | 10% |

The text replay component is important.

The objective is:

\[
\text{add vision}
\quad \textbf{without destroying}
\quad
\text{text System-One ability}.
\]

If catastrophic text regression appears, raise replay to 20–30%.

---

# 10. Candidate data construction

## Native multiple-choice data

Use the original candidates whenever available.

Examples include datasets such as:

- A-OKVQA;
- ScienceQA;
- IconQA;
- AI2D.

A-OKVQA natively provides four candidate choices and the correct candidate index.

## Open-answer datasets

Do not blindly convert all open-answer VQA into multiple choice.

Only generate candidate sets when hard negatives can be constructed reliably.

Good negatives should be:

```text
same semantic type
similar plausibility
same approximate length
not obviously absurd
```

Bad:

```text
Question: What animal is visible?

cat
dog
carburetor
democracy
```

Good:

```text
cat
dog
fox
wolf
```

Otherwise the model learns candidate artifacts rather than visual decision-making.

---

# 11. License strategy

Maintain:

```text
data/manifests/sources.jsonl
```

with:

```text
source
version
original_url
license
image_license
redistribution_allowed
commercial_use
split
checksum
```

This is essential.

For example, ScienceQA's licensing information specifies CC BY-NC-SA restrictions, and IconQA is also non-commercial.

Therefore maintain two possible dataset manifests:

```text
decision_mix_research.json
decision_mix_permissive.json
```

Do not automatically publish an Apache-2.0-trained artifact after combining non-commercial datasets without resolving the licensing implications.

---

# 12. Split protocol

Do **not** random-split individual questions.

Split by the underlying information source.

For images:

```text
same image → exactly one split
```

For video:

```text
same original video → exactly one split
```

For UI trajectories:

```text
same episode / website / workflow → exactly one split
```

For robot trajectories:

```text
same episode → exactly one split
```

Use four partitions:

```text
Train          80%
Model Val       8%
Calibration     6%
Test            6%
```

The calibration split must never influence gradient updates.

Additionally reserve entire datasets/task families as OOD evaluation.

Recommended OOD-only candidates include:

```text
MMStar
MMMU
RealWorldQA
MMBench
```

depending on contamination and licensing considerations.

---

# 13. Leakage removal

Before training:

### Image deduplication

Use:

```text
SHA256
pHash
```

to detect exact and visually near-duplicate images.

### Text deduplication

Use normalized text plus:

```text
MinHash / SimHash
```

over:

```text
state
question
candidate descriptions
```

### Group leakage

Remove any example whose source task/episode appears in test.

### Benchmark contamination

Never mix benchmark validation/test material into training merely because labels have become publicly available.

---

# 14. Multimodal leakage protocol

This must be a first-class evaluation component.

Multimodal benchmarks can accidentally reveal visually relevant information in text, allowing models to solve supposedly visual questions without using the image. This has already been demonstrated in multimodal safety evaluation.

Therefore every important evaluation should include four conditions:

```text
FULL
image + text

TEXT-ONLY
text, image removed

IMAGE-SHUFFLED
correct text + unrelated matched image

IMAGE-ONLY
image + minimal question
```

For image-essential subsets, you want:

\[
Acc_{FULL} \gg Acc_{TEXT}
\]

and:

\[
P_y(I_{correct})
>
P_y(I_{shuffled}).
\]

Otherwise you have built a textual classifier with decorative images.

---

# 15. Training objective

For question \(q\) containing \(K\) candidates:

\[
\mathbf{s}
=
(s_1,\ldots,s_K)
\]

and:

\[
\mathbf{p}=
\operatorname{softmax}(\mathbf{s}).
\]

## Primary loss — listwise cross entropy

\[
L_{CE}=-\log p_y.
\]

Compute losses **per question**, not per candidate.

Otherwise questions with more candidates receive more weight.

---

## Calibration auxiliary — Brier loss

\[
L_{Brier}
=
\sum_{i=1}^{K}
(p_i-\mathbb{1}[i=y])^2.
\]

Recommended initial objective:

\[
\boxed{
L=L_{CE}+\lambda_B L_{Brier}
}
\]

with:

\[
\lambda_B=1.
\]

Then ablate:

```text
λB ∈ {0, 0.25, 1.0}
```

OneJev independently uses cross-entropy plus an equal-weight Brier component, making this a useful comparable baseline.

Do not optimize ECE directly.

ECE is an evaluation statistic, not an ideal differentiable training objective.

---

# 16. Optional multimodal grounding loss

Only use this on examples with verified counterfactual images.

For correct image \(I\) and known-invalid counterfactual \(I'\):

\[
L_{ground}
=
\max
\left[
0,
m-
(s_y(I)-s_y(I'))
\right].
\]

Then:

\[
L=
L_{CE}
+\lambda_B L_{Brier}
+\lambda_G L_{ground}.
\]

Start with:

```text
λG = 0.1
margin = 0.5
```

Do not apply this loss to arbitrary shuffled natural images because the new image might accidentally remain compatible with the answer.

---

# 17. Training stages

The experiment should be progressive.

The scientific question is not:

> Can full fine-tuning make the model work?

It almost certainly can.

The useful question is:

> **What is the minimum adaptation required?**

---

## V0 — frozen VLM/direct logits

No training.

Evaluate:

### V0-A
Answer-letter logits.

### V0-B
Length-normalized candidate sequence likelihood.

### V0-C
Generative JSON choice.

Measure:

```text
accuracy
NLL
Brier
ECE
latency
candidate-order sensitivity
```

This establishes how much specialization is actually needed.

---

## V1 — frozen VLM + decision head

Train:

```text
scalar decision head
```

Freeze:

```text
vision tower
vision merger
language model
token embeddings
```

This is the most important low-cost experiment.

If V1 already performs strongly, you have evidence that Qwen's multimodal representation contains enough information and only requires a learned decision surface.

### Suggested configuration

```text
head                 Linear(2560, 1)

optimizer            AdamW
LR                    3e-4
weight decay          0.01
warmup                3%
epochs                2–5
gradient clipping     1.0
precision             BF16
```

Select checkpoint using:

\[
ValScore=
NLL+0.25\,Brier.
\]

Not test accuracy.

---

# 18. V2 — language LoRA + head

This should be the **default candidate for your final model**.

Train:

```text
language LoRA
decision head
```

Freeze:

```text
entire vision tower
vision merger initially
```

Recommended LoRA:

```text
rank          32
alpha         64
dropout       0–0.05
bias          none
```

Apply to language-model linear transformations:

```text
full-attention projections
DeltaNet projections
MLP projections
```

Do not use an unrestricted `"all-linear"` selector unless you verify that it excludes the vision tower.

### Learning rates

```text
LoRA              5e-5 – 1e-4
decision head     1e-4 – 3e-4
```

### Training

```text
epochs                    1–3
effective batch           64 questions
scheduler                 cosine
warmup                    3%
gradient checkpointing    enabled
grad norm                 1.0
precision                 BF16
```

For ≥100K examples, start with LoRA dropout 0.

For small training sets, use approximately 0.05.

---

# 19. V3 — language LoRA + vision merger + head

Train:

```text
language LoRA
vision patch merger
decision head
```

Keep the 24 visual Transformer blocks frozen.

Qwen's visual merger maps the vision representation into the language hidden space, making it the lowest-cost place to modify visual-language alignment.

Suggested LR:

```text
head            1e-4
LoRA            5e-5
vision merger   1e-5
```

Use V3 only if V2 shows:

```text
strong text performance
but
weak image-essential performance
```

or poor robustness under visual distribution shift.

---

# 20. V4 — upper vision blocks

Train:

```text
top 4 vision blocks
vision merger
language LoRA
decision head
```

Freeze lower 20 visual blocks.

Suggested LR:

```text
vision blocks   3e-6 – 1e-5
merger          1e-5
LoRA            3e-5 – 5e-5
head            1e-4
```

V4 tests whether task-specific visual adaptation is truly necessary.

It should not be run simply because more training usually improves performance.

---

# 21. V5 — full VLM fine-tuning

Train:

```text
vision encoder
vision merger
language model
decision head
```

Use this only as an upper-bound experiment.

Recommended LR:

```text
1e-6 – 5e-6
```

with:

```text
FSDP or ZeRO-3
BF16
gradient checkpointing
low-resolution curriculum
```

V5 should answer:

> What performance remains unavailable to PEFT?

not:

> What should we deploy?

---

# 22. Image processing

Use the official `AutoProcessor`.

Do not replace Qwen's multimodal positional encoding or image-token arrangement. Hugging Face explicitly warns that Qwen3.5 uses multimodal RoPE across temporal/height/width dimensions.

The default preprocessing configuration permits very large images, so training should enforce an explicit pixel budget.

Recommended initial training:

```text
ordinary images/screens
≤ approximately 1M pixels

OCR/doc examples
selectively allow approximately 2M pixels
```

Bucket batches by:

```text
text length
visual token count
number of images
```

rather than ordinary text-token count alone.

---

# 23. Image augmentation

Avoid generic computer-vision augmentation pipelines.

For screenshots, diagrams and documents:

```text
horizontal flip = invalid
large crops = often invalid
rotation = often invalid
```

Allowed mild augmentation:

```text
JPEG compression
small brightness shift
small contrast shift
resolution degradation
minor blur
```

Use stronger corruptions for robustness evaluation rather than normal training.

---

# 24. Long context

Do not immediately train with 16K tokens.

Use curriculum:

```text
Stage 1
≤ 2K–4K text tokens

Stage 2
≤ 8K

Stage 3 optional
≤ 16K
```

Long states should be deliberately included near the end if the target application needs full-page/UI/document contexts.

Qwen3.5 itself supports a far longer native context; training every decision at that length would simply waste compute.

---

# 25. Text replay

For every multimodal optimizer step distribution, retain a fraction of original text System-One training data.

For example:

```text
80% multimodal
20% text
```

Measure:

\[
\Delta_{text}
=
Acc_{after}-Acc_{before}.
\]

Target:

```text
text regression < 1 absolute percentage point
```

unless multimodal improvement clearly justifies otherwise.

This makes the project genuinely:

> text decision model → multimodal decision model

instead of:

> new multimodal model trained independently.

---

# 26. Calibration

Calibration must happen **after model training**.

MLLMs are known to exhibit substantial miscalibration, and post-hoc temperature scaling remains a strong baseline.

Freeze all model parameters.

Optimize \(T>0\):

\[
T^*
=
\arg\min_T
-\sum_n
\log
P_{T}(y_n\mid x_n)
\]

using only:

```text
Calibration split
```

Never validation or test.

Evaluate:

```text
global T
```

first.

Then, if enough calibration data exist:

```text
T_choice
T_noul
T_score
```

can be learned separately.

Only use per-type temperatures if each type has enough held-out observations.

---

# 27. Recalibration after quantization

If final serving uses:

```text
INT8
INT4
GGUF
AWQ
GPTQ
```

do:

```text
train
→ merge adapters if applicable
→ quantize
→ recalibrate
→ final test
```

not:

```text
train
→ calibrate
→ quantize
→ ship
```

because quantization may alter the score distribution.

---

# 28. Evaluation metrics

## Decision quality

Report:

```text
accuracy
macro-F1
per-task accuracy
per-source accuracy
```

For imbalanced binary decisions:

```text
AUROC
AUPRC
```

---

## Probability quality

Report at minimum:

```text
NLL
multiclass Brier score
ECE
adaptive ECE
```

Also report:

```text
reliability diagrams
calibration slope/intercept
```

---

## Selective prediction

Calculate risk-coverage curves.

At confidence threshold \(\tau\):

\[
Coverage(\tau)
=
\frac{
|\{x:\max_i p_i(x)\ge \tau\}|
}{N}.
\]

\[
Risk(\tau)=1-Accuracy_{\tau}.
\]

Report:

```text
AURC
accuracy @ 90% coverage
accuracy @ 75% coverage
coverage at target error rate
```

This matters more operationally than ECE alone.

---

# 29. Visual grounding metrics

Define:

### Text-only gap

\[
G_{text}
=
Acc_{full}-Acc_{text-only}.
\]

### Correct-image confidence gain

\[
G_{img}
=
P_y(I_{correct})
-
P_y(I_{shuffled}).
\]

### Image corruption robustness

Evaluate:

```text
original
JPEG
blur
downsample
occlusion
brightness
```

Plot accuracy and calibration versus corruption severity.

A System-One model that stays accurate but becomes badly overconfident under blur is not reliable.

---

# 30. Candidate robustness

Even with pointwise scoring, test:

### Candidate permutation

Randomly permute candidate presentation order.

Expected:

\[
\Delta P\approx 0.
\]

### Candidate paraphrase

Replace:

```text
"click submit"
```

with:

```text
"press the submit control"
```

and measure decision stability.

### Candidate count

Evaluate:

```text
K = 2
4
8
16
32
```

even if training primarily uses 2–8 candidates.

This determines whether calibration transfers as candidate-set size changes.

---

# 31. OOD evaluation

Evaluate three forms separately.

## Domain OOD

Train on:

```text
natural images
documents
UI
```

hold one entire family out.

## Visual OOD

Unseen:

```text
styles
resolutions
camera domains
screen themes
```

## Decision OOD

Known images but previously unseen question/candidate semantics.

This distinguishes:

```text
visual generalization
```

from:

```text
decision-interface generalization.
```

---

# 32. Baselines

The final experiment should contain at least:

```text
B0 Qwen3.5-4B generative choice
B1 Qwen3.5-4B answer-token logits
B2 Qwen3.5-4B candidate sequence likelihood
B3 V1 frozen VLM + head
B4 V2 language LoRA + head
B5 V3 LoRA + visual merger
B6 V4 upper-vision adaptation
B7 V5 full VLM upper bound
```

And current external System-One comparisons where reproducibly available:

```text
OneJev-4B
OmniJev-4B
relevant open text System-One scorer
```

OneJev explicitly reports Qwen3.5 multimodal decision fine-tuning with the vision tower frozen, one epoch, \(5\times10^{-6}\) LR and CE+Brier; OmniJev reports Qwen3.5 LoRA plus decision/ordinal heads and post-hoc calibration.

Therefore those are more scientifically meaningful comparisons than a generic LLaVA baseline.

---

# 33. Statistical protocol

Use at least three random seeds for the trainable variants selected for final comparison.

Report:

```text
mean ± standard deviation
```

and 95% bootstrap confidence intervals.

For paired classification comparisons:

```text
McNemar test
```

can be used.

For:

```text
Brier
NLL
AURC
latency
```

use paired bootstrap confidence intervals.

Do not claim improvement from a 0.3-point change without uncertainty analysis.

---

# 34. Efficiency evaluation

Since this is a System-One model, inference efficiency is part of the research claim.

Measure:

```text
p50 latency
p95 latency
p99 latency
peak VRAM
questions / second
candidates / second
```

for:

```text
1 question
4 questions
16 questions
```

and varying candidate count.

Also compare:

```text
decision forward
vs
generative Qwen decoding
```

at identical state/image input.

---

# 35. Prefix-branch serving

For several questions sharing one image/state:

```text
image + shared state
      ↓
shared prefill/cache
      ↓
 ┌────┼────┬────┐
 q1c1 q1c2 q2c1 q2c2 ...
      ↓
batched candidate scoring
```

Do not repeatedly encode the image for every option in production.

This architecture is already used by recent System-One implementations because it amortizes the expensive shared state across many decisions.

For V1/V2 where the vision tower is frozen, visual embeddings can additionally be cached during training if storage permits.

---

# 36. Training implementation

Recommended core stack:

```text
PyTorch
Transformers
PEFT
Accelerate/FSDP
custom grouped-question trainer
```

Do not use ordinary `Trainer` classification loss directly because the model receives candidate branches individually but loss must be normalized jointly across candidates belonging to one question.

Each batch item needs:

```text
question_id
candidate_index
gold_candidate_index
candidate_count
task_type
source
```

Pseudo-code:

```python
scores = model(branch_batch).logits.squeeze(-1)

loss = 0

for group in group_by_question(scores):
    logits = group.scores
    probs = softmax(logits, dim=-1)

    ce = -log(probs[group.gold])
    brier = ((probs - one_hot(group.gold)) ** 2).sum()

    loss += ce + lambda_brier * brier

loss /= number_of_questions
```

---

# 37. Fast Qwen kernels

Qwen3.5 uses a hybrid stack containing Gated DeltaNet.

Transformers notes that without the optimized linear-attention dependencies the model can fall back to slower, more memory-intensive PyTorch operations.

Therefore benchmark:

```text
causal_conv1d / FLA-enabled
vs
fallback
```

on the actual GPU environment before final latency reporting.

Record which kernel implementation was used in the experiment metadata.

---

# 38. Recommended resource path

For a 4B/approximately-5B-total multimodal model:

| Stage | Recommended hardware |
|---|---|
| V0 | 1×24 GB |
| V1 | 1×24 GB |
| V2 LoRA | 1×24 GB feasible; 2×24 GB preferable |
| V3 merger + LoRA | 2×24 GB recommended |
| V4 vision adaptation | ≥2×24 GB, preferably larger |
| V5 full VLM | larger-memory multi-GPU preferred |

For V2 on constrained memory use:

```text
gradient checkpointing
small microbatch
BF16
8-bit optimizer
pixel caps
gradient accumulation
```

QLoRA can be evaluated as an engineering alternative, but I would make ordinary BF16 LoRA the scientific reference where feasible.

---

# 39. Model-selection rule

Do not automatically choose the most heavily fine-tuned model.

Use a Pareto criterion.

A version is better only if it improves:

\[
\{
Accuracy,
Calibration,
OOD,
VisualGrounding
\}
\]

without disproportionate degradation in:

\[
\{
Latency,
Memory,
TextRetention
\}.
\]

If V2 matches V4/V5 within statistical uncertainty, choose V2.

That itself is an important result.

---

# 40. Expected likely final architecture

My strongest prior is that the final practical model will be:

```text
Qwen/Qwen3.5-4B
        │
        ├── frozen vision encoder
        │
        ├── frozen or lightly tuned vision merger
        │
        ├── rank-32 language LoRA
        │
        └── scalar candidate head
                ↓
candidate logits
                ↓
temperature scaling
                ↓
typed probabilities
```

rather than full VLM fine-tuning.

In notation:

\[
\boxed{
P(c_i|I,S,Q,C)
=
\operatorname{softmax}_i
\left(
\frac{
w^\top
f_{\theta+\Delta\theta_{LoRA}}
(I,S,Q,c_i)
+b
}{T}
\right)
}
\]

with the vision encoder initially frozen.

---

# 41. Core ablation table

The minimum scientific ablation should be:

| Variant | Head | Language | Merger | Vision |
|---|---|---|---|---|
| V0 | none | frozen | frozen | frozen |
| V1 | train | frozen | frozen | frozen |
| V2 | train | LoRA | frozen | frozen |
| V3 | train | LoRA | train | frozen |
| V4 | train | LoRA | train | top-4 |
| V5 | train | full | full | full |

For every row measure:

```text
ID accuracy
OOD accuracy
NLL
Brier
ECE
AURC
text retention
visual reliance
latency
VRAM
```

This table is essentially the backbone of the eventual paper.

---

# 42. Main research questions

The resulting research should be organized around:

### RQ1 — Transferability

Can a text-trained System-One decision mechanism be transferred to image-conditioned decisions without redesigning its decision interface?

### RQ2 — Adaptation depth

How much of a pretrained multimodal model must be updated to obtain strong visual decision accuracy?

### RQ3 — Calibration

Do calibrated decision probabilities remain reliable under visual distribution shift and image corruption?

### RQ4 — Modality dependence

Does the resulting model actually use visual evidence rather than textual shortcuts?

### RQ5 — Efficiency

What accuracy/calibration improvement is obtained per unit of trainable parameters and inference cost?

---

# 43. Hypotheses

### H1

A frozen multimodal Qwen representation with a learned decision head will substantially outperform raw generative decision logits.

### H2

Language-side LoRA will provide most of the improvement required to transform the VLM into a decision model.

### H3

Updating the visual merger or vision tower will improve image-intensive and OOD tasks but produce diminishing returns relative to computational cost.

### H4

Post-hoc temperature scaling will substantially improve calibration without changing top-1 accuracy.

### H5

Training with text replay will allow multimodal adaptation while preserving the original System-One text decision capability.

---

# 44. Important negative result to test

One particularly valuable result would be:

\[
V1 \approx V2 \approx V5
\]

on decision accuracy.

That would mean the visual model already contains the necessary representation and only needs a decision readout.

Conversely:

\[
V1 < V2 < V3 < V4
\]

would demonstrate that increasingly deep multimodal adaptation is required.

Both outcomes are scientifically meaningful.

---

# 45. Scientific novelty warning

As of late September/early October 2026, the claim:

> “We introduce a multimodal Qwen3.5 System-One model that returns typed calibrated probabilities without generating text.”

is **not sufficient novelty**.

OneJev already reports multimodal Qwen3.5 System-One models trained with frozen vision and CE+Brier, and OmniJev reports Qwen3.5 multimodal decision models using LoRA, decision heads, ordinal outputs and calibration.

Therefore the research contribution should instead be framed around:

\[
\boxed{
\text{Text-to-Multimodal Decision Transfer}
}
\]

plus:

\[
\boxed{
\text{minimum adaptation depth}
}
\]

and preferably:

\[
\boxed{
\text{multimodal calibration under modality shift}.
}
\]

That is much more defensible.

---

# 46. Final recommended experiment sequence

Run in this order:

```text
Text System-One winner
        ↓
V0 frozen Qwen multimodal baselines
        ↓
V1 frozen VLM + transferred/new head
        ↓
V2 language LoRA + head
        ↓
calibrate + full evaluation
        ↓
Does visual grounding remain weak?
        │
        ├── NO → stop
        │
        └── YES
             ↓
            V3 merger
             ↓
            V4 upper vision
             ↓
            V5 upper-bound only
```

The stopping rule is important.

Do not spend the compute for V4/V5 unless the evidence says visual representation adaptation is the bottleneck.

---

# 47. Recommended final model-card fields

## Status

```text
DESIGNED — training not started
```

## Base model

```text
Primary:
Qwen/Qwen3.5-4B

Ablation:
Qwen/Qwen3.5-4B-Base

Revision:
immutable Hugging Face commit SHA recorded in ARTIFACTS.lock
```

## Architecture

```text
(image(s), state, question, candidate)
→ Qwen3.5 multimodal cross-encoder
→ final non-padding hidden state
→ shared candidate scoring head
→ grouped candidate softmax
→ calibrated typed decision probabilities
```

## Autoregressive?

```text
Backbone: causal/autoregressive architecture.

Decision inference: non-autoregressive.
No answer tokens are generated.
```

## Decision probability

\[
p_i=\operatorname{softmax}(s_i/T)
\]

with \(T\) fit only on a held-out calibration set.

## Training objective

\[
L=L_{CE}+\lambda_B L_{Brier}
\]

with optional validated counterfactual visual-grounding loss.

## Training stages

```text
V0 direct frozen baseline
V1 frozen VLM + head
V2 language LoRA + head
V3 merger + language LoRA + head
V4 upper vision blocks + merger + LoRA + head
V5 full VLM upper bound
```

## Recommended deployment candidate

```text
V2 unless V3/V4 provides statistically significant
visual/OOD gains.
```

## Primary evaluation

```text
Decision accuracy
NLL
Brier
ECE
AURC
candidate invariance
visual reliance
OOD robustness
text retention
latency
VRAM
```

## Core scientific question

> How efficiently can a validated text System-One decision mechanism be transferred to image-grounded decision making, and what multimodal adaptation is actually necessary for accurate and calibrated decisions?