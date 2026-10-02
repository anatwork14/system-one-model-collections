# Metrics

Use accuracy/top-1 for choice correctness; NLL and Brier for probability quality; ECE with documented bins for calibration; AURC for selective prediction where abstention/confidence is meaningful. Report option-count and domain slices. Softmax normalization is only a numerical property, not a calibration guarantee.
