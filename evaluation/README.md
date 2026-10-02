# Evaluation protocol

`run_all.py --list` inventories registered methods. To execute inference, an adapter must map a method's native output into `{choice_id: probability}` and return a record with model/checkpoint revisions, tokens, option count, latency, memory, success, and notes. The common runner rejects malformed distributions but does not interpret missing probability values as zeros.

The released smoke case in `datasets/released_smoke.jsonl` is infrastructure validation only. It is not a benchmark and must not be used to tune prompts or thresholds. Report accuracy, NLL, Brier, ECE, and AURC for labeled evaluation sets when applicable. Softmax normalization alone is not calibration.

Never compare `UPSTREAM_*` and `OURS_*` results without displaying that identity in the table. Record raw per-example outputs and a dataset manifest/hash alongside aggregates.
