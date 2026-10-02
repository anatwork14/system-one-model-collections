# Frozen configs

`full_ft.yaml` is the prepared single-device Kev trainer recipe with host-offloaded FP32 masters. `fsdp.yaml` records optional multi-rank mixed-precision and sharding settings; the Kev trainer initializes FSDP2 under `torchrun`. Review resource fit before any run. Full FT is intentionally last in the run order.
