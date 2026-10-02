# Backbone module mapping

The release contains a merged Qwen3.5-9B checkpoint. A parameter-by-parameter delta audit needs a separate unmodified `Qwen/Qwen3.5-9B` checkpoint at the matching base revision. That checkpoint is not downloaded because the requested installation is limited to Clef-Flash.

The local Qwen3.5-4B weight index confirms language projections in full-attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`), MLP layers (`gate_proj`, `up_proj`, `down_proj`), and linear-attention layers (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`). These names define the explicit local LoRA target set in `retrain/15_clef_style_4b/src/train.py`. This mapping is a reproduction choice, not evidence of Cloudflare's targets.

`scripts/compare_backbone.py` accepts two local model directories and writes `audit/parameter_diff.csv`; it does not download either checkpoint. Do not use the current 4B base for a 9B delta audit.
