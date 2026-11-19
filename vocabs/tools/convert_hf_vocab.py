#!/usr/bin/env python3
"""Convert a HuggingFace ByteLevel-BPE tokenizer into a `.tiktoken.zst` oracle.

This is how the HF-sourced vocabularies are produced (llama3, qwen2,
deepseek_v3, mistral_v3, the 2026-08 additions glm4 / glm5 / minimax_m2, and
the 2026-11 additions qwen3_5 / llama4 / mistral_v15). mistral_v15 is read from
Mistral's `tekken.json` instead of an HF tokenizer.json. Kimi needs no conversion — Moonshot ships a native tiktoken `tiktoken.model`,
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
    pip install tokenizers tiktoken zstandard mistral-common
    python vocabs/tools/convert_hf_vocab.py qwen3_5 llama4 mistral_v15
    python vocabs/tools/build_tkv.py        qwen3_5 llama4 mistral_v15
"""

import base64
import json
import pathlib
import sys

import tiktoken as pytiktoken
import zstandard
from tokenizers import Tokenizer

TESTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "tiktoken" / "tests"
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
    # Qwen 3.5 through 3.8 share this tokenizer byte for byte. It is a new
    # vocabulary, not an extension of qwen2's.
    "qwen3_5": (
        "Qwen/Qwen3.5-27B",
        248_044,
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ),
    # ungated mirror of meta-llama/Llama-4-Scout-17B-16E-Instruct
    "llama4": (
        "unsloth/Llama-4-Scout-17B-16E-Instruct",
        200_000,
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ),
}

# Tekken vocabularies ship as `tekken.json`, not an HF tokenizer.json. Ids
# below num_special are control tokens; regular rank r has id r + num_special.
# The oracle stores the control tokens as literal bytes at their ids, the same
# layout as the mistral_v3 oracle converted from Nemo's tokenizer.json.
TEKKEN_SOURCES = {
    "mistral_v15": ("mistralai/Mistral-Large-4-1T-A52B", 204_800),
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
    write_oracle(name, ranks)


def convert_tekken(name: str):
    """Self-checked against mistral-common's Tekkenizer, the reference
    implementation, rather than against tiktoken with the same pattern, which
    would only compare the conversion with itself."""
    from huggingface_hub import hf_hub_download
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer

    repo, want_vocab = TEKKEN_SOURCES[name]
    path = hf_hub_download(repo, "tekken.json")
    data = json.loads(pathlib.Path(path).read_text())
    cfg = data["config"]
    n_special = cfg["default_num_special_tokens"]
    if cfg["default_vocab_size"] != want_vocab:
        raise SystemExit(f"{name}: {repo} vocab {cfg['default_vocab_size']} != expected {want_vocab}")

    ranks = {st["token_str"].encode(): st["rank"] for st in data["special_tokens"][:n_special]}
    for entry in data["vocab"][: want_vocab - n_special]:
        ranks[base64.b64decode(entry["token_bytes"])] = entry["rank"] + n_special
    if sorted(ranks.values()) != list(range(want_vocab)):
        raise SystemExit(f"{name}: ranks are not exactly 0..{want_vocab}")

    ref = Tekkenizer.from_file(path)
    regular = {tk: r for tk, r in ranks.items() if r >= n_special}
    enc = pytiktoken.Encoding(name=name, pat_str=cfg["pattern"], mergeable_ranks=regular, special_tokens={})
    checked = 0
    for text in corpus():
        ours = enc.encode(text, disallowed_special=())
        theirs = ref.encode(text, bos=False, eos=False)
        if ours != theirs:
            raise SystemExit(f"{name}: divergence on {text!r}\n  converted: {ours}\n  tekken:    {theirs}")
        checked += 1
    print(f"{name}: self-check OK ({checked} texts)", file=sys.stderr)
    write_oracle(name, ranks)


def write_oracle(name: str, ranks: dict):
    lines = b"".join(
        base64.b64encode(tk) + b" " + str(rank).encode() + b"\n"
        for tk, rank in sorted(ranks.items(), key=lambda kv: kv[1])
    )
    out = OUT_DIR / f"{name}.tiktoken.zst"
    out.write_bytes(zstandard.ZstdCompressor(level=19).compress(lines))
    print(f"{name}: wrote {out} ({out.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    names = sys.argv[1:] or [*SOURCES, *TEKKEN_SOURCES]
    for n in names:
        if n in TEKKEN_SOURCES:
            convert_tekken(n)
        else:
            convert(n)
