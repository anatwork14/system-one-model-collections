# Clef-Flash joint head inspection

Source inspected: pinned `joint_schema_model.py`, `joint_head_config.json`, and `joint_head.safetensors` in `weights/architecture/`.

## Structure found in released code

| Component | Published implementation |
|---|---|
| Head class | `JointSchemaHead` |
| Backbone input width | 4,096 (the config's 9B hidden size) |
| Internal width | 1,024 |
| Evidence routing | 2 `EvidenceRoutingLayer` blocks; option queries attend to projected sequence states |
| Cross-field reasoning | 4 `TransformerDecoderLayer` blocks; fields attend to the state memory |
| Attention heads | 16 in both head blocks |
| Feedforward width | 4,096 |
| State representation | Final normalized backbone states, projected as token memory; final token also supplies a global vector |
| Question representation | Mean of hidden states over encoded question-instruction span, then projected and combined with question-type embedding and option summary |
| Option representation | Mean hidden states over encoded option span, plus mean output-embedding vectors for option token IDs |
| Lexical prior | Normalized output-embedding option vectors dot a normalized question-plus-global anchor, scaled by learned `prior_logit_scale` |
| Cross-question interaction | Question/field vectors are processed jointly by the 4-layer transformer decoder |
| Scorer | Learned cosine joint score plus a residual MLP over field, option, product, and absolute-difference features; added to lexical prior |
| Output | Variable-length list of one logit tensor per question; each tensor has one value per valid option |
| Released tensor count | 122 tensors, 121,762,820 parameters; see `head_tensor_shapes.txt` |

## 4B port implication

The current local Qwen3.5-4B config reports hidden size 2,560. Six input projections and the `hidden_norm` depend on backbone hidden size; their 4,096-wide checkpoint tensors cannot be loaded into the 4B head. The local arm changes only the constructor's `hidden_size` and initializes a fresh head. Internal 1,024-wide routing and decoder layers keep their published architecture and shapes but are also freshly initialized, avoiding any implicit partial transfer from the 9B trained head.

The published code is reused as the source implementation. Training, the 4B adaptation, dataset, and any RL stage are local reproduction choices. The exact Cloudflare LoRA target-module list and training coefficients are not available in the release.
