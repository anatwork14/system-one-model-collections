# Open-world protocol

Planned partitions: `known_train`, `known_test`, `novel_supplied`, and `novel_unsupplied`. Known classes occur during training. Novel-supplied classes are absent from training but their class descriptions are provided at inference. Novel-unsupplied omits the correct class and requires `UNKNOWN`. Keep classes and source families disjoint across partitions; never leak novel class labels into prompt tuning.
