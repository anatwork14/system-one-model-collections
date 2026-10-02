# Benchmark registry

The full benchmark catalog, source locks, dataset cards, group manifests, and download/validation tooling live in [`benchmarks/`](../../benchmarks/README.md). This workspace directory remains the home for benchmark definitions that are shared by method evaluations.

Each frozen manifest must identify source revision, license, split, row IDs, evaluation protocol, and metrics. The released smoke fixture is intentionally stored under `evaluation/datasets/` and is not a quality benchmark.
