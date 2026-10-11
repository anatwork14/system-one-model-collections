# M16 research gap and revalidation notes

The proposal's broad direction is plausible, but its novelty framing needed revision before implementation.

## Verified prior art as of 6 October 2026

- Qwen3.5-4B has a 32-layer text decoder, hidden size 2,560, and a hybrid linear/full-attention layout. These are direct fit for layerwise probes at four-layer structural intervals. [Official Qwen3.5-4B configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- OD-1 Base includes a layer-8 self-exit. Its model card reports validation accuracy 0.589 from that exit alone, and 0.760 accuracy at threshold 0.99 after 0.997 of requests escalated. This supports the proposal's warning that a shallow confidence threshold may save little compute. [OD-1 Base model card](https://huggingface.co/mvbalaji/od1-base)
- RSI-Jev v5.0-VL uses the first 20 layers of Qwen3.5-4B-Base, supports typed text/image decisions, and reports its quality curve flattened by layer 20. Static Qwen truncation and multimodal typed decision serving are therefore existing baselines, not M16 novelty. [RSI-Jev v5.0-VL model card](https://huggingface.co/shgao/rsi-jev-v5.0-vl-3b)
- Clef already uses a joint schema head over a Qwen prefill, jointly scoring allowed typed options. M16 should reuse the M15 interface and keep the study focused on depth. [Cloudflare Clef release](https://blog.cloudflare.com/clef-decision-models/)
- Generic dynamic LLM exits have recent work, including FlexEE and LEDE. M16 must not claim the first adaptive-depth LLM, early-exit Transformer, or learned halting method. [FlexEE](https://arxiv.org/abs/2609.17008), [LEDE](https://arxiv.org/abs/2606.03113)

## Protocol correction

The proposal originally used the 324-row outer evaluation set to choose exit positions. That conflicts with this workspace's frozen-test policy and would contaminate final evaluation. P0 now uses only the existing 2,268/408 train/dev split. The outer set must remain unopened through probe training, checkpoint selection, and exit selection.

The proposal also described P0 as training M15-style probes but did not specify their readout size. The implemented P0 makes that explicit: identical 256-wide projection and 128-wide scorer probes over pooled question, option, and global-state representations. Since this differs from the M15 joint schema head, later controlled comparisons must use one shared decision readout when comparing depth policies.

## Defensible research question

After P0: do per-request risk features and cross-exit stability enable better quality/latency tradeoffs than static 20-layer truncation, confidence thresholds, and a full-depth M15-style model under the same data, readout, hardware, and serving conditions?

This is a candidate research question. It is not a novelty or performance result until multi-exit training, true staged execution, matched baselines, and leakage-safe evaluation are complete.
