#!/usr/bin/env python3
"""Inspect candidate tokenization without loading model weights or using a GPU."""
import argparse
from transformers import AutoTokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", help="local tokenizer directory or pinned Hub model")
    p.add_argument("tokens", nargs="*", default=list("ABCD"))
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    failed = False
    for token in a.tokens:
        ids = tok.encode(token, add_special_tokens=False)
        print(f"{token!r}\t{ids}\t{'single-token' if len(ids) == 1 else 'MULTI-TOKEN'}")
        failed |= len(ids) != 1
    if failed:
        raise SystemExit("At least one candidate label is not a single token; use the method's actual tokenizer-aware scoring path.")

if __name__ == "__main__": main()
