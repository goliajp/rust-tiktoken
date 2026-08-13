# Vocabulary payload decomposition — 2026-08-13

Working notes behind the 4.0.0 data format. Kept in the repository (excluded
from the published package) so the rejected branches do not get re-measured.

## Baseline

```
cargo package --no-verify (3.8.3)   10,437,850 B / 10,485,760 B cap = 99.54%
12 .tiktoken.zst files               10,325,280 B = 98.9% of the package
tiktoken_wasm_bg.wasm                11,449,576 B (10,727,163 gzipped)
```

17 encodings share 12 data files (`gpt2`→r50k, `kimi_k3`→kimi_k2,
`deepseek_v4`→deepseek_v3, `o200k_harmony`→o200k, `p50k_edit`→p50k).

The shipped form was `zstd("<base64(token bytes)> <rank>\n" × N)`.

## Candidates measured

All on the same 12 vocabularies, zstd `-19` unless noted, byte totals:

| Candidate | Total | Verdict |
|---|--:|---|
| F0 shipped, base64 + rank column | 10,325,280 | — |
| Higher zstd level (`-22 --ultra --long=27`) | 2 B better | rejected: the shipped files were already `-19`-equivalent (same text recompressed: 632,884 vs 632,887 shipped) |
| Trained zstd dictionary, 110 KB / 1 MB | 6,634,054 / 6,645,228 | rejected: larger than no dictionary |
| o200k as a raw content dictionary for the rest | 6,100,087 | rejected: saves 415 KB but forces decompressing 1.6 MB of o200k to load any other vocabulary |
| All 12 concatenated into one frame | 4,379,939 | not reachable: no random access. Recorded as the cross-vocabulary redundancy bound |
| F2 length block ‖ byte block | 6,515,290 | kept |
| F3 u8 lengths with escape | 6,515,393 | rejected: no better than varint, one more special case |
| **F8 = F2 with the body grouped by length class** | **6,280,896** | **kept — free** (the length block is read first, so the decoder already knows the class) |
| F9 F8 + lexicographic order within class + permutation | 8,798,470 | rejected: the permutation costs more than the ordering saves |
| F10 whole-vocabulary sort + front-coding + permutation | 6,891,892 | rejected: zstd already exploits shared prefixes better than explicit front-coding |
| F5 BPE merge pairs (left, right) of earlier ranks | 6,049,325 | rejected: 3.7% better than F8 for an index build, a second code path, and 1,059 non-decomposable escapes |
| F8 payload under brotli `-q 11 -w 24` | 5,837,464 | rejected: 7% better for a new decoder dependency; also 5.8 ms vs zstd's faster decode, and decoder code in the wasm artifact |
| F8 payload under xz `-9e` | 5,933,644 | rejected: 26.3 ms to decode, 5x zstd |

## Rank-aligned extensions

Comparing vocabularies rank by rank (token at rank `i` identical in both):

| | shares a prefix of | which is |
|---|--:|---|
| llama3 vs cl100k_base | 100,256 | all of cl100k_base (llama3 = cl100k + 27,744) |
| glm5 vs glm4 | 151,329 | all of glm4 (glm5 = glm4 + 3,491) |
| p50k_base vs r50k_base | 50,256 | all of r50k_base (p50k = r50k + 24) |

Every other pair shares at most a few hundred leading ranks. qwen2 and glm4
overlap 0.68 by token set but share only 7,948 leading ranks, so only a
dictionary could exploit it — and a qwen-only user would download 578 KB of
glm4 to save 380 KB, which is a loss for them.

Storing only the tail for those three: **−1,133,669 B**.

## Result

```
vocabulary payload   10,325,280 -> 5,148,835   (-50.1%)
.crate               10,437,850 -> 5,269,228   (99.5% -> 50.2% of the cap)
wasm, all 17         11,449,576 -> 6,267,165   (10,727,163 -> 5,546,760 gzipped)
wasm, o200k only                   1,917,753   (1,202,806 gzipped)
```

Headroom is now ~5.2 MB, about eight more vocabularies at current sizes.

## What this does not solve

- **The `.crate` download.** Cargo fetches the whole package regardless of
  features, so every dependent still downloads 5.3 MB. Only moving the data
  into separate optional crates fixes that, and it is also the only thing that
  raises the cap again. Trigger to revisit: **package over 8 MB (80% of the
  cap)**. `tiktoken-vocab`, `tiktoken-vocabs`, `tiktoken-data` and
  `tiktoken-encodings` were all unclaimed on crates.io as of 2026-08-13.
- **The wasm artifact still ships every vocabulary it was built with.** Loading
  vocabulary data over the network on demand would take the browser cost to
  ~1.1 MB of code plus one vocabulary, but it makes `getEncoding` async — a
  breaking change for the npm package, and it needs a public constructor taking
  raw vocabulary bytes (`CoreBpe::new` is `pub(crate)`).
- **crates.io will raise the cap per crate on request** (help@crates.io). Not
  worth asking while half the payload was still base64 padding.
