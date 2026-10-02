# Pointer-head-only ablation

Keep Kev's pointer serialization and head implementation as the reference. Freeze every backbone tensor. Train only the pointer head on the shared Nimble-derived train rows. Compare with LoRA+pointer using identical initialization, rows, seed, and evaluation protocol.
