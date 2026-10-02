# Results policy

`released/released_smoke.csv` currently contains the schema header only. The released smoke scripts are prepared, but no inference has run. This sheet is reserved for same-input infrastructure checks, not a benchmark. `controlled/` stores local reproduction runs. `tables/` stores aggregate exports and metric definitions. A missing result is not a failure and must remain distinguishable from a failed run.

Use result IDs `UPSTREAM_semif_q35_4b`, `UPSTREAM_simple_jev_q35_4b`, `UPSTREAM_tev1_q35_4b`, `UPSTREAM_jevk5_q35_4b`, `UPSTREAM_kev_q35_4b`, `OURS_nimble_q35_4b_seed17`, `OURS_clm_q35_4b_seed17`, `OURS_pointeronly_q35_4b_seed17`, and `OURS_fullft_pointer_q35_4b_seed17`. Include the exact source/model/data revisions in every result row.
