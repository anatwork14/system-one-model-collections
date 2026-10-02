# Kev decision data

The released base recipe `decision-v7` contains 10,000 examples from ten public datasets, 896 generated policy examples, and 1,680 examples from 60 generated rule structures. The current Kev-4B checkpoint adds a later skills delta: 11,320 new records plus 4,000 replayed `decision-v7` records. The 6,000 `hard-v1` labels are programmatically generated; 5,320 `devtools-v1` items derive from CodeReviewer, CommitPackFT, FlakeFlagger, and Aegis (with source label semantics varying by dataset).

The public sources named in the model card include Banking77, BoolQ, AG News, MultiNLI, SST-5, Yelp Review Full, TREC, DBpedia-14, Amazon Reviews Multi, and IMDB. Exact licenses/provenance are listed in the pinned upstream model card and per-dataset manifests; retain those records. Generated labels are not human ground truth. Do not train or evaluate from a mixed directory without first separating the intended train/development/test partitions.
