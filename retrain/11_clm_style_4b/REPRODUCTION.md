# Controlled CLM-style reproduction

This is a small controlled adaptation, not a reproduction of CLM's 91M-scale training. Use only the Nimble train split, convert the correct candidate to a positive state/action pair, and treat remaining candidates as negatives. Freeze the base encoder and fit projections/temperature. Record pair-generation code and hashes.
