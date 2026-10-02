# Download status

Pinned source snapshots, shared base checkpoints, and all five published released checkpoints are present. Nimble's 2,676 train / 324 frozen evaluation records and Tev1's published `new-v1` train/development recipe are materialized with content hashes. Exact source commits, Hub revisions, file hashes, and data manifests are recorded in the adjacent locks and `DOWNLOAD_STATUS.json`.

No training is needed for the current request, and no training has been run. Model inference, target-host resource validation, and performance measurement have not been run.

Cloudflare Clef-Flash is pinned at Hub revision `17f0b0ad64efb65d273590632833508766b2aae6`. The complete official 9B model is downloaded and verified: all four weight shards match the pinned Hub file sizes, and all 760 indexed tensors match the safetensors index and its tensor byte total. SHA-256 hashes for the architecture bundle and full release are recorded in `released/05_clef_flash_9b/weights/MANIFEST.json`. The local Clef-style 4B head shape check passes. Any 4B training is deferred; it is not part of the current setup.

Prepared at: 2026-10-01T16:38:57+00:00
