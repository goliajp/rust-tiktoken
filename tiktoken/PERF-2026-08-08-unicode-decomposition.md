# Unicode path decomposition — vs gpt-tokenizer 3.4.0

Phase A (read-only). Trigger: a browser benchmark for the website found
`gpt-tokenizer`, a **pure-JavaScript** tokenizer, beating this crate on
Unicode-dense text. Not a wasm artefact — native Rust loses too.

## Measured baseline

Apple M-series laptop, all figures from one machine so the ratios are
apples-to-apples. Harness: 50-iteration warmup, median of 9 timed rounds,
identical corpora on both sides (generator reproduced byte-for-byte in
`tiktoken/examples` and `web/src/bench.ts`). Token counts verified equal
across this crate, `gpt-tokenizer` and `js-tiktoken` before timing — the three
do the same work.

`unicode_varied_4kb` — 3,868 B, 267 pieces, 2,492 tokens, no repetition:

| stage | ours (native) | gpt-tokenizer (JS) | ratio |
| --- | ---: | ---: | ---: |
| S1 pre-tokenize (regex) | 19.3 µs | 15.4 µs | 1.25× |
| S2 whole-piece lookup | 2.1 µs | — | — |
| S3 BPE merge | 48.9 µs | 37.4 µs | 1.31× |
| **total** | **70.3 µs** | **52.8 µs** | **1.33×** |
| ours via wasm | 110.4 µs | | 2.09× |

Stage sum reconciles to the standalone measurement within 4%.

`unicode_4kb` — the same sentence ×50:

| stage | ours (native) | gpt-tokenizer | ratio |
| --- | ---: | ---: | ---: |
| S1 pre-tokenize | 16.6 µs | 14.6 µs | 1.14× |
| S2+S3 | 65.4 µs | 25.0 µs | 2.62× |
| **total** | **82.0 µs** | **39.6 µs** | **2.07×** |

The gap widens on repetitive input because `gpt-tokenizer` memoises whole
pre-token pieces in a 100k-entry LRU (`BytePairEncodingCore.ts:466`,
`mergeCache`), which a ×50 corpus turns into a ~98% hit rate. Repetition is
an amplifier, not the cause: the gap is still 1.33× with no repetition at all.

For contrast, ASCII is not affected — `long_45kb` is 60.0 µs for us against
442 µs for `gpt-tokenizer`. Every piece hits the vocabulary whole, so S3 is
zero. **The hand-written ASCII scanner was never the problem; the Unicode
path simply never got the same attention.**

## Where S3 goes

Instrumented `Vocab::get` with a call counter and a key-length histogram
(probes reverted; see git history of this commit's parent for the patch).

`unicode_varied_4kb`:

| | |
| --- | --- |
| vocab lookups during merge | 6,089 (2.4 per token) |
| time per lookup | **8.0 ns** (48.9 µs ÷ 6,089) |
| lookups that **miss** | 4,699 — **77%** |
| keys ≤ 8 bytes | **96.7%** (2 B: 59%, 3 B: 25%, 4 B: 8%) |

6,089 × 8.0 ns = 48.7 µs, i.e. **S3 is entirely vocabulary lookup latency**.
The linked-list bookkeeping, the min-scan and the array memmoves together
account for under 0.5 µs. Any attack on the merge *algorithm* is attacking
0.4% of the cost.

## Why a lookup costs 8 ns

`vocab.rs` — open addressing, `Slot { rank: u32, offset: u32, len: u16,
occupied: bool }` = 12 B, table sized to `2 × entries` rounded up. For
cl100k_base that is 262,144 slots = **3.1 MB**, plus a ~470 KB arena.

`Vocab::get` (`vocab.rs:132`) does, per probe step:

1. `fx_hash(token)` — cheap.
2. `self.table[idx]` — random access into 3.1 MB. Misses L2.
3. `self.arena[slot.offset..]` then `stored == token` — **a second random
   access into a different 470 KB region**, on every occupied slot visited,
   including slots that do not match.

So the common case is two dependent cache misses, and the 77% of lookups that
miss pay the arena access for every colliding slot they walk past without
ever needing the bytes.

8 ns is about right for that shape on this hardware; it is not a constant
factor to shave, it is one memory access too many.

## Attack candidates

| # | change | file | expected | risk |
| --- | --- | --- | ---: | --- |
| 1 | Inline keys ≤ 8 B into the slot; keep the arena offset only for longer tokens (union on the 8 bytes). Removes access #3 for 96.7% of merge lookups. | `vocab.rs:17` (layout), `:132` (`get`) | the dominant win — targets ~97% of the 8 ns | slot grows 12 B → 16 B, table 3.1 → 4.2 MB; must re-measure, a bigger table is a worse first access |
| 2 | Store a 1-byte tag from the high hash bits in the slot; reject a non-matching slot without comparing bytes. Cuts the arena access out of the 77% miss path even for long keys. | `vocab.rs:17`, `:132` | complements #1; alone it fixes misses only | free — the byte is already lost to padding |
| 3 | `bpe_count` allocates a `Vec<usize>` per piece purely to return `len() - 1`. Give the merge a count-only path. | `merge.rs:243` | 267 allocations for this corpus; small next to #1 but free | none, internal |
| 4 | Memoise piece → token count / ids, as `gpt-tokenizer` does. Closes the 2.6× on repetitive input. | new | large on repeated text (chat templates, system prompts) | needs interior mutability on a `&'static` shared across threads — a lock would cost more than it saves at 8 ns. Needs its own design, probably a per-thread cache. Do **not** bundle with #1–#3. |
| 5 | Pre-tokenize is 1.25× slower than V8's Irregexp on the same pattern. | `pretokenize.rs` | 19.3 → ~15 µs at best | separate attack surface; only worth it after S3 lands, when it becomes the majority stage |

Order: **2 → 1 → 3** as one change to the vocabulary layout, measured
together (each alone is likely inside the noise band), then re-decompose
before touching #4 or #5.

## Gates

- **Pre-Phase-A** — gap confirmed on median-of-3 runs, ±4% run-to-run, on
  identical corpora, on one machine. Not a single-run misread.
- **Pre-Phase-B** — S3 is 69.5% of self-time, far above the 10 pp bar, and it
  decomposes to a single named cost (lookup latency × count) rather than to a
  hand-waved estimate.

## What this does not claim

`gpt-tokenizer` is faster than this crate only on Unicode-dense input. On
ASCII it is 7.4× slower, and `js-tiktoken` is 30–60× slower than this crate
everywhere. The finding is narrow and specific: **our Unicode merge path is
memory-bound and one indirection too deep.**
