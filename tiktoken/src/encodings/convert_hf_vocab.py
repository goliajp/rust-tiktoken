#!/usr/bin/env python3
"""Convert a HuggingFace ByteLevel-BPE tokenizer into a `.tiktoken.zst` oracle.

This is how the HF-sourced vocabularies are produced (llama3, qwen2,
deepseek_v3, mistral_v3, and the 2026-08 additions glm4 / glm5 / minimax_m2).
Kimi needs no conversion — Moonshot ships a native tiktoken `tiktoken.model`,
which only needs zstd compression.

Output lands in `tests/vocab-oracle/`, which is the reference form and is *not*
part of the published package. Run `build_tkv.py` afterwards to produce the
`.tkv.zst` file this crate actually embeds; the `vocab_oracle` tests in
`src/encoding.rs` diff the two so the compact form can never drift from the
reference.

The HF `tokenizer.json` stores its vocabulary as GPT-2 byte-unicode mapped
strings; each entry is reverse-mapped to raw bytes and written as the tiktoken
line format `<base64(token bytes)> <rank>`, then zstd-compressed. Added tokens
are NOT part of the base vocabulary — they are special tokens, registered in
`encoding.rs`.

Every conversion is self-checked before writing: the converted ranks plus the
tokenizer's own split regex are loaded into a Python `tiktoken.Encoding` and
differentially compared against the HF tokenizer over the parity corpus
(`tests/canonical_corpus.py`), skipping texts that contain an added token.

Usage:
    pip install tokenizers tiktoken zstandard
    python tiktoken/src/encodings/convert_hf_vocab.py glm4 glm5 minimax_m2
    python tiktoken/src/encodings/build_tkv.py    glm4 glm5 minimax_m2
"""

import base64
import json
import pathlib
import sys

import tiktoken as pytiktoken
import zstandard
from tokenizers import Tokenizer

TESTS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "tests"
OUT_DIR = TESTS_DIR / "vocab-oracle"
sys.path.insert(0, str(TESTS_DIR))
from canonical_corpus import corpus  # noqa: E402

# encoding name -> (reference repo, expected base vocab size, split regex for
# the self-check — the oniguruma-compatible pattern from the tokenizer.json,
# converted where needed for Python `regex`)
SOURCES = {
    "glm4": (
        "zai-org/GLM-4.5",
        151_329,
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ),
    "glm5": (
        "zai-org/GLM-5.2",
        154_820,
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ),
    "minimax_m2": (
        "MiniMaxAI/MiniMax-M2",
        200_000,
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ),
}


def bytes_to_unicode():
    """GPT-2's byte -> printable-unicode table (mirrors openai/gpt-2)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


UNICODE_TO_BYTE = {v: k for k, v in bytes_to_unicode().items()}


def hf_token_to_bytes(token: str) -> bytes:
    return bytes(UNICODE_TO_BYTE[ch] for ch in token)


def convert(name: str):
    repo, want_vocab, pat = SOURCES[name]
    tok = Tokenizer.from_pretrained(repo)
    data = json.loads(tok.to_str())
    vocab = data["model"]["vocab"]
    if len(vocab) != want_vocab:
        raise SystemExit(f"{name}: {repo} vocab {len(vocab)} != expected {want_vocab}")

    ranks = {hf_token_to_bytes(tk): rank for tk, rank in vocab.items()}
    assert len(ranks) == len(vocab), "byte-mapping collision"

    # --- self-check: converted ranks must reproduce HF token ids exactly ---
    enc = pytiktoken.Encoding(name=name, pat_str=pat, mergeable_ranks=ranks, special_tokens={})
    added = {a["content"] for a in data.get("added_tokens", [])}
    checked = skipped = 0
    for text in corpus():
        if any(a in text for a in added):
            skipped += 1
            continue
        ours = enc.encode(text, disallowed_special=())
        hf = tok.encode(text, add_special_tokens=False).ids
        if ours != hf:
            raise SystemExit(f"{name}: divergence on {text!r}\n  converted: {ours}\n  hf:        {hf}")
        checked += 1
    print(f"{name}: self-check OK ({checked} texts, {skipped} skipped)", file=sys.stderr)

    lines = b"".join(
        base64.b64encode(tk) + b" " + str(rank).encode() + b"\n"
        for tk, rank in sorted(ranks.items(), key=lambda kv: kv[1])
    )
    out = OUT_DIR / f"{name}.tiktoken.zst"
    out.write_bytes(zstandard.ZstdCompressor(level=19).compress(lines))
    print(f"{name}: wrote {out} ({out.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    names = sys.argv[1:] or list(SOURCES)
    for n in names:
        convert(n)
