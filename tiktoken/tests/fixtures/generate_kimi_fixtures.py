#!/usr/bin/env python3
"""Regenerate the kimi_k2 / kimi_k3 oracle fixtures from Moonshot's reference.

Kimi has no HuggingFace `tokenizer.json` — Moonshot ships a native tiktoken
vocabulary (`tiktoken.model`) driven by `tokenization_kimi.py`. The reference
implementation is therefore a Python `tiktoken.Encoding` built from that file
and the `pat_str` defined there, which is exactly what this script constructs.

K2 and K3 share a byte-identical vocabulary; only the special-token tables
differ, and special tokens don't affect plain-text encoding — so one fixture
corpus serves both encodings and this script writes kimi_k2.json + kimi_k3.json
with identical content, keeping the one-fixture-per-encoding convention.

Usage:
    pip install tiktoken huggingface_hub
    python tiktoken/tests/fixtures/generate_kimi_fixtures.py
"""

import base64
import json
import pathlib
import sys

import tiktoken as pytiktoken
from huggingface_hub import hf_hub_download

OUT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))
from generate_hf_fixtures import corpus  # noqa: E402  (same corpus as the HF fixtures)

REPO = "moonshotai/Kimi-K3"  # byte-identical tiktoken.model to Kimi-K2-Instruct
EXPECTED_VOCAB = 163_584

# pat_str from tokenization_kimi.py (identical in K2 and K3)
PAT_STR = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)


def load_ranks() -> dict[bytes, int]:
    path = hf_hub_download(REPO, "tiktoken.model")
    ranks = {}
    with open(path, "rb") as f:
        for line in f:
            tok, rank = line.split()
            ranks[base64.b64decode(tok)] = int(rank)
    if len(ranks) != EXPECTED_VOCAB:
        raise SystemExit(f"vocab {len(ranks)} != expected {EXPECTED_VOCAB} — reference drifted")
    return ranks


def main():
    enc = pytiktoken.Encoding(
        name="kimi", pat_str=PAT_STR, mergeable_ranks=load_ranks(), special_tokens={}
    )
    cases = []
    for text in corpus():
        tokens = enc.encode(text, disallowed_special=())
        cases.append({"text": text, "tokens": tokens, "count": len(tokens)})

    payload = json.dumps(cases, indent=2, ensure_ascii=False) + "\n"
    for name in ("kimi_k2", "kimi_k3"):
        (OUT_DIR / f"{name}.json").write_text(payload)
        print(f"wrote {len(cases)} cases to {OUT_DIR / f'{name}.json'}")


if __name__ == "__main__":
    main()
