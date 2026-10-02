# Kev-style full fine-tuning upper bound

This arm uses the same candidate serialization and pointer objective as the PEFT arm but updates every backbone parameter. Keep it last in the run order. Freeze configuration, data IDs, and evaluation before launching; record any sharding/offload behavior in the result metadata.
