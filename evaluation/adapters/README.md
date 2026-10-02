# Method adapters

Implement one adapter per native inference interface. Adapter output schema is defined in `evaluation/README.md`. Preserve raw native response beside normalized probabilities. Record failures rather than dropping malformed responses.
