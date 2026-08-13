#!/usr/bin/env python3
"""Build the shipped `.tkv.zst` vocabulary files from the `.tiktoken.zst` oracles.

Pipeline for a vocabulary:

    upstream (HF / tiktoken.model)
      -> convert_hf_vocab.py -> tests/vocab-oracle/<name>.tiktoken.zst   (base64 text, not packaged)
      -> build_tkv.py        -> src/encodings/<name>.tkv.zst             (binary, packaged)

The oracle keeps the human-inspectable `<base64(token)> <rank>` form and stays
out of the published crate (`exclude = ["tests/"]`); the `vocab_oracle` tests in
`src/encoding.rs` diff every shipped `.tkv.zst` against it, so the two can never
drift.

TKV1 frame layout (the whole frame is one zstd stream):

    "TKV1"                 4 B    magic
    n_tokens               4 B    u32 LE
    rank0                  4 B    u32 LE   rank of this frame's first token
    lengths                varint x n_tokens, in rank order
    body                   token bytes, grouped by length class (ascending),
                           each class in rank order

Two properties earn their keep over the old `<base64> <rank>` text form:

  - no base64 (33% inflation that also hides byte-level repeats from zstd) and
    no rank column (ranks are consecutive, so `rank0 + i` reconstructs them);
  - regrouping the body by length class costs nothing — the length block is in
    rank order, so the decoder already knows which class to draw from — and
    compresses better because same-length tokens share structure.

Three vocabularies are exact rank-aligned extensions of another (verified here
before writing, and again by those tests): the derived file holds only the tail,
and the loader concatenates base frame + delta frame. Nothing is recomputed at
run time — a derived vocabulary's first N entries *are* the base's.

Usage:
    pip install zstandard
    python tiktoken/src/encodings/build_tkv.py            # all
    python tiktoken/src/encodings/build_tkv.py glm5       # one
"""

import base64
import pathlib
import struct
import sys
from collections import defaultdict

import zstandard

OUT_DIR = pathlib.Path(__file__).resolve().parent
ORACLE_DIR = OUT_DIR.parent.parent / "tests" / "vocab-oracle"

MAGIC = b"TKV1"
ZSTD_LEVEL = 19

# derived vocabulary -> base vocabulary (rank-aligned prefix extension)
DERIVED = {
    "llama3": "cl100k_base",
    "glm5": "glm4",
    "p50k_base": "r50k_base",
}


def read_oracle(name: str) -> list[tuple[bytes, int]]:
    """Parse `<base64(token)> <rank>` lines into (token, rank) in rank order."""
    raw = zstandard.ZstdDecompressor().decompress(
        (ORACLE_DIR / f"{name}.tiktoken.zst").read_bytes(), max_output_size=64 << 20
    )
    out = []
    for line in raw.split(b"\n"):
        if not line:
            continue
        token, rank = line.rsplit(b" ", 1)
        out.append((base64.b64decode(token), int(rank)))
    return out


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        n |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return n, pos
        shift += 7


def encode_frame(entries: list[tuple[bytes, int]]) -> bytes:
    tokens = [t for t, _ in entries]
    for i, (_, rank) in enumerate(entries):
        if rank != entries[0][1] + i:
            raise SystemExit(f"non-consecutive rank at index {i}: {rank}")
    classes = defaultdict(list)
    for token in tokens:
        classes[len(token)].append(token)
    body = b"".join(b"".join(classes[k]) for k in sorted(classes))
    lengths = b"".join(varint(len(t)) for t in tokens)
    header = MAGIC + struct.pack("<II", len(tokens), entries[0][1])
    return zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(header + lengths + body)


def decode_frame(frame: bytes) -> list[tuple[bytes, int]]:
    """Reference decoder — mirrors `parse_tkv` in encoding.rs, used for the self-check."""
    buf = zstandard.ZstdDecompressor().decompress(frame, max_output_size=64 << 20)
    if buf[:4] != MAGIC:
        raise SystemExit("bad magic")
    n_tokens, rank0 = struct.unpack("<II", buf[4:12])
    pos = 12
    lengths = []
    for _ in range(n_tokens):
        length, pos = read_varint(buf, pos)
        lengths.append(length)
    counts = defaultdict(int)
    for length in lengths:
        counts[length] += 1
    offsets, cursor = {}, pos
    for length in sorted(counts):
        offsets[length] = cursor
        cursor += length * counts[length]
    if cursor != len(buf):
        raise SystemExit(f"body length mismatch: {cursor} != {len(buf)}")
    out = []
    for i, length in enumerate(lengths):
        out.append((buf[offsets[length] : offsets[length] + length], rank0 + i))
        offsets[length] += length
    return out


def build(name: str) -> tuple[int, int]:
    entries = read_oracle(name)
    if name in DERIVED:
        base = read_oracle(DERIVED[name])
        if entries[: len(base)] != base:
            raise SystemExit(f"{name} is not a rank-aligned extension of {DERIVED[name]}")
        payload = entries[len(base) :]
    else:
        payload = entries

    frame = encode_frame(payload)
    if decode_frame(frame) != payload:
        raise SystemExit(f"{name}: round-trip mismatch")

    out = OUT_DIR / f"{name}.tkv.zst"
    out.write_bytes(frame)
    old = (ORACLE_DIR / f"{name}.tiktoken.zst").stat().st_size
    print(
        f"{name}: {len(entries)} tokens"
        + (f" ({len(payload)} beyond {DERIVED[name]})" if name in DERIVED else "")
        + f" -> {out.name} {len(frame)} bytes (was {old}, {100 * len(frame) / old - 100:+.0f}%)",
        file=sys.stderr,
    )
    return old, len(frame)


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.name[: -len(".tiktoken.zst")] for p in ORACLE_DIR.glob("*.tiktoken.zst"))
    total_old = total_new = 0
    for n in names:
        old, new = build(n)
        total_old += old
        total_new += new
    print(
        f"total: {total_old} -> {total_new} ({100 * total_new / total_old - 100:+.0f}%)",
        file=sys.stderr,
    )
