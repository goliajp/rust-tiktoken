# Changelog

All notable changes to this crate / npm package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.3.0] - 2026-11-24

Version lockstep with `tiktoken` 5.3.0, which adds `CoreBpe::memo` / `TextMemo`
on the Rust side. The wasm package itself is unchanged.

## [5.2.1] - 2026-11-23

Version lockstep with `tiktoken` 5.2.1, a documentation release: benchmark
tables for 5.2, and a comparison with the fastest alternatives.

## [5.2.0] - 2026-11-23

Version lockstep with `tiktoken` 5.2.0: its regex-free pre-tokenizers, new
vocabulary lookup and faster merges, and its change for 336 code points on
`mistral_v15`, `deepseek_v3` and `deepseek_v4` (see that crate's changelog).
The JavaScript API is unchanged. `count` and `encode` are substantially
faster; updated benchmark tables will follow in a later release. The wasm
build has no threads, so the native crate's automatic parallel encoding does
not apply here.

### Changed
- **Built with WebAssembly SIMD (`simd128`)**: the pre-tokenizer classifies
  text 64 bytes at a time. The package now needs Chrome 91, Firefox 89 or
  Safari 16.4 and later; older browsers fail to compile the module.
- Text reaches the tokenizer in a buffer kept across calls: through
  `TextEncoder.encodeInto`, or `TextEncoder.encode` plus a copy where that
  is faster (Safari; timed once on first use), or, when most of the text is
  outside ASCII, as UTF-16 code units converted to UTF-8 inside the module.
  Ids come back with one copy. The generated bindings copied ASCII one char
  at a time and allocated on every call. `count` on a string of a few chars
  makes one more call between JavaScript and WebAssembly than before.
- Where the engine has the JS string builtins (`wasm:js-string`; current
  Chrome, Safari and Node have them), text mostly outside ASCII is copied
  into the module by `intoCharCodeArray` instead of a JavaScript loop, and
  eight 3-byte chars are converted to UTF-8 at a time. Other engines keep
  the loop.
- Each split pattern is compiled into a function of its own, and the module
  is built without reference types, so that Safari no longer leaves the
  tokenizer in its baseline tier.
- Encodings with the same ranks (`o200k_base` and `o200k_harmony`,
  `p50k_base` and `p50k_edit`, `deepseek_v3` and `deepseek_v4`, `kimi_k2`
  and `kimi_k3`) share one vocabulary, and the lookup tables are sized to
  the vocabulary instead of the next power of two. All 20 encodings loaded
  and used take 197 MiB of linear memory (5.1.1: 196 MiB).
- The `.wasm` with all 20 vocabularies is 8.65 MB, 8.01 MB gzipped (5.1.1:
  8.96 MB, 8.22 MB); an o200k-only build is 1.64 MB, 1.00 MB gzipped (5.1.1:
  1.96 MB, 1.22 MB). The `regex` crate is no longer compiled in.
- `encode`, `encodeWithSpecialTokens`, `count` and `countWithSpecialTokens`
  throw a `TypeError` for an argument that is not a string.
- The npm package carries the bindings' `snippets/` directory.

## [5.1.1] - 2026-11-21

Documentation only, in lockstep with `tiktoken` 5.1.1.

### Fixed
- The npm README said "17 encodings" (the Japanese one "11"); it is 20.
- The size of an o200k-only build is re-measured: 1.96 MB, 1.22 MB gzipped
  (was quoted as 1.92 / 1.20 MB from 4.0).
- The model-name example and API docs list all eight tokenizing vendors, and
  the build commands include `--scope goliapkg`.

## [5.1.0] - 2026-11-20

Version lockstep with `tiktoken` 5.1.0, whose vocabulary data now comes from
per-vendor data crates. The JavaScript API, the token ids and the npm package
contents are unchanged; a custom wasm build that selects only some
vocabularies now downloads only those vendors' data crates.

## [5.0.1] - 2026-11-20

Version lockstep with `tiktoken` 5.0.1. No API change.

### Fixed
- **The npm package shipped without any license file**, though the wasm
  embeds every vocabulary. It now carries LICENSE-MIT, LICENSE-APACHE,
  LICENSE-3RD-PARTY, NOTICE and the full third-party license texts in
  `licenses/`; the crate on crates.io carries the same files.
- The stack-overflow fix and the completed third-party licenses from
  `tiktoken` 5.0.1.

## [5.0.0] - 2026-11-20

Version lockstep with `tiktoken` 5.0.0. The JavaScript API keeps every function
and signature; what changes is the data behind it.

### Added
- **Three encodings**: `llama4` (Llama 4, Meta Muse), `qwen3_5` (Qwen 3.5 –
  3.8) and `mistral_v15` (Mistral Large 4). `listEncodings()` returns 20 names,
  and `getEncoding` accepts the new ones. Each has its `vocab-*` feature for a
  custom build.
- **Pricing for 153 models** (was 116): GPT-6, Claude 5.x, Gemini 4, Qwen 4,
  Kimi K3.1, GLM-5.3, MiniMax M3 and more; see the `tiktoken` 5.0.0 entry.

### Changed
- **`modelToEncoding` / `encodingForModel` return a different encoding** for
  Llama 4 (`llama4`, was `llama3`), Qwen 3.5 – 3.8 (`qwen3_5`, was `qwen2`),
  Mistral Large 4 and `mistral-large-latest` (`mistral_v15`, was
  `mistral_v3`) and `deepseek-flash` (`deepseek_v4`), so their token counts
  change. `gpt-6*` now resolves to `o200k_base`, which OpenAI has not
  confirmed.
- **The artifact grows by the three vocabularies.**

  | | 4.0.0 | 5.0.0 |
  |:--|--:|--:|
  | `tiktoken_wasm_bg.wasm` | 6,277,462 | 8,963,334 |
  | gzip -9 | 5,550,867 | 8,217,849 |

  4.0.0 is the published npm artifact; 5.0.0 is measured on a local
  `wasm-pack build --target web --release`. A build with only the vocabularies you use stays
  small; see the README.

## [4.0.1] - 2026-08-13

Version lockstep with `tiktoken` 4.0.1. The JavaScript API, the token ids and
the published artifact are all unchanged — the default build carries every
vocabulary, and that is the one on npm.

### Fixed
- **A build that selects a subset of vocabularies no longer emits dead-code
  warnings.** The `vocab-*` features exist so a wasm bundle can carry only what
  it needs, but taking that path surfaced 13 warnings out of `tiktoken` for the
  vocabularies left behind — measured on
  `--no-default-features --features vocab-o200k_base`, now 0. Nothing about the
  default build changes; see the `tiktoken` 4.0.1 entry for why.

## [4.0.0] - 2026-08-13

Version lockstep with `tiktoken` 4.0.0. The JavaScript API is unchanged — same
functions, same signatures, same token ids, still every encoding in one
package.

### Changed
- **The artifact is 45% smaller.** `tiktoken` 4.0.0 re-encodes its vocabulary
  data (base64 and the redundant rank column dropped; llama3, glm5 and
  p50k_base stored as tails of the vocabularies they extend), and ~90% of this
  package is that data.

  | | 3.8.3 | 4.0.0 |
  |:--|--:|--:|
  | `tiktoken_wasm_bg.wasm` | 11,449,576 | 6,277,462 |
  | gzipped, over the wire | 10,727,163 | 5,550,867 |

  Both are the published npm artifact. The remaining ~1.1 MB that is not
  vocabulary data is code and the pricing tables.

### Added
- **`vocab-*` features, forwarded from `tiktoken`.** Building this crate
  yourself, you can carry only the vocabularies you need:

  ```sh
  wasm-pack build --target web --release -- \
      --no-default-features --features vocab-o200k_base
  ```

  measures 1,917,753 bytes (1,202,806 gzipped) — an 8.9x smaller download than
  3.8.3 for a page that only counts GPT-4o/GPT-5 tokens. The published npm
  package continues to carry all 17 vocabularies.

## [3.8.3] - 2026-08-10

Version lockstep with `tiktoken` 3.8.3 (token-id upgrade documentation on the
crate side); no behavior change in the wasm package.

## [3.8.2] - 2026-08-09

### Docs
- The README's "Supported Models" table was a hand-copied listing frozen at
  the 3.5-era catalogue; the npm page was missing the gpt-5 family, the
  Moonshot/Zhipu/MiniMax providers and DeepSeek V4 entirely. Now a
  per-provider summary (107 models / 10 providers) with `allModels()` and
  docs.rs as the authority. No code changes.

## [3.8.1] - 2026-08-09

Version lockstep with `tiktoken` 3.8.1 (README corrections on the crate side);
no behavior change in the wasm package.

## [3.8.0] - 2026-08-09

Inherits `tiktoken` 3.8.0 — see the [crate changelog](../tiktoken/CHANGELOG.md)
for the o200k `[\r\n/]*` fix and the full performance notes.

### Changed
- **2–4x faster than gpt-tokenizer 3.4 in the browser on every corpus, CJK
  prose included** (previously 2–3x slower on CJK): layered vocabulary lookup,
  CJK fast-path scanners, and whole-piece memoisation land in the wasm build.
- The wasm artifact now optimizes for speed (`opt-level = 3`, `wasm-opt -O3`
  instead of `-Os`): ~97% of the file is vocabulary data, so the faster code
  section costs ~130 KB against 11.3 MB.
- Reproducible in-browser benchmark against gpt-tokenizer and js-tiktoken:
  `npm run bench` in the repository's `web/`.

## [3.7.1] - 2026-08-08

Dependency refresh in lockstep with `tiktoken` 3.7.1 (`base64` 0.23, `ruzstd`
0.9); no behavior change — vocabulary loading re-verified against the full
reference differential corpora on the new versions.

## [3.7.0] - 2026-08-08

Inherits `tiktoken` 3.7.0 — see the [crate changelog](../tiktoken/CHANGELOG.md).

### Added
- Six new encodings, focused on the Chinese open-weights ecosystem: `kimi_k2` /
  `kimi_k3` (Moonshot), `glm4` / `glm5` (Zhipu), `minimax_m2` (MiniMax), and
  `deepseek_v4` — all verified byte-exact against their reference tokenizers.
  `encodingForModel` resolves `kimi-*`, `glm-*`, `minimax-*`, `deepseek-v4*`,
  and the `deepseek-chat` / `deepseek-reasoner` API aliases.
- Pricing for 13 models across the new Moonshot / Zhipu / MiniMax providers
  (**107 models, 10 providers** total); `modelsByProvider` accepts the new
  provider names.

### Changed
- The wasm binary grows by ~4.4 MB (compressed vocabularies for the new
  encodings are embedded, as with all encodings).

## [3.6.0] - 2026-08-08

Inherits the `tiktoken` 3.6.0 fixes — see the
[crate changelog](../tiktoken/CHANGELOG.md) for full detail.

### Fixed
- `cl100k_base` and `o200k_base` no longer split canonical newline tokens
  ([#5](https://github.com/goliajp/rust-tiktoken/issues/5)). `encode("word\n\nnext")`
  now returns the canonical `[1178, 271, 3684]` rather than
  `[1178, 198, 198, 3684]`. Verified byte-identical to `openai/tiktoken` 0.13.0
  over a 10,491-case corpus. **Token ids change for newline-containing text** —
  recompute any cached counts or stored id sequences.
- `mistral_v3` now uses Tekken's own pre-tokenizer pattern instead of the
  cl100k stand-in it shared before; `deepseek_v3` no longer splits whitespace
  runs that precede a digit or CJK character. Both changed token ids for the
  affected text.
- `qwen2` and `deepseek_v3` gained the added tokens they were missing —
  `qwen2`'s `<tool_call>` / `<|fim_*|>` / `<|repo_name|>` / `<|file_sep|>` and
  `deepseek_v3`'s `<｜User｜>` / `<｜Assistant｜>` / tool markers / 800
  placeholders. `encodeWithSpecialTokens` could not produce these ids before.
- `encodingForModel` resolves the legacy `davinci-codex`, `*-edit-001`, and
  first-generation `text-search-*` / `text-similarity-*` / `code-search-*` model
  ids to the same encodings as upstream.

### Internal
- Repaired the native-target test suite, which was silently broken and not in
  CI: `list_encodings_count` still asserted 9 encodings (stale since 3.3.0 made
  it 11) and four error-path tests panic on non-wasm targets because
  `JsValue` construction aborts there — those are now gated to `wasm32`.

### Added
- Pricing data covers **94 models** across 7 providers (was 68), including the
  OpenAI GPT-5.6 family, Anthropic's Claude 5 generation, `gemini-3.6-flash`,
  `deepseek-v4-*`, `qwen3.8-max`, and Mistral's Devstral 2 / Ministral 3 SKUs.

## [3.5.1] - 2026-06-07

### Changed
- Dual-licensed under **MIT OR Apache-2.0** (previously MIT only):
  `LICENSE-MIT` + `LICENSE-APACHE`, `license = "MIT OR Apache-2.0"`.
- `tiktoken` path-dep bumped to 3.5.1.

## [3.5.0] - 2026-06-07

> From this release, `tiktoken` and `tiktoken-wasm` share one version number and
> are released in lockstep (this crate jumps 3.4.0 → 3.5.0 to match `tiktoken`).

### Changed
- `tiktoken` path-dep bumped to 3.5.0. Inherits the ASCII fast-path
  pre-tokenizers (cl100k / o200k / qwen2 / deepseek — 2.3–5.5x faster ASCII
  `encode`/`count`), the hybrid linear/heap BPE merge, the chunked two-pass
  `encode_parallel`, and pricing for Claude Opus 4.7/4.8 + the Gemini 3 series
  (68 models total).

## [3.4.0] - 2026-06-07

### Changed
- `tiktoken` path-dep bumped to 3.3.0. Inherits:
  - gpt2 registered as the 11th encoding (alias for r50k_base).
  - Six OpenAI GPT-5.x model entries (gpt-5/gpt-5-mini/gpt-5-nano/gpt-5.4/
    gpt-5.4-mini/gpt-5.5) with Standard/Batch/Flex/Priority and long-context
    tier data where verified.
  - VisionPricing rewritten as a provider-specific enum
    (OpenAI tile-based, OpenAI patch-based, Anthropic divisor, Gemini tile).
- Doc comment lists `o200k_harmony`, `gpt2` as supported encoding names.
- No WASM API changes.

## [3.3.0] - 2026-06-06

### Changed
- `tiktoken` path-dep bumped to 3.2.0. Inherits o200k_harmony encoding
  support (the new gpt-oss family route), the extended pricing schema
  (batch / high-tier / audio / vision), and the Meta llama price
  re-sourcing.
- Doc comment on `getEncoding` lists `o200k_harmony` as a supported
  encoding name.
- No WASM API changes — `listEncodings()` returns the new entry
  automatically.

## [3.2.5] - 2026-06-06

### Changed
- `tiktoken` path-dep bumped to 3.1.5. Inherits the underlying lib's
  `model_to_encoding` improvements (gpt-5 / gpt-4.5 / ft: / Azure prefix support,
  davinci-002 / babbage-002 routing fix) and refreshed 2026-06 pricing data.
- No WASM API changes.

## [3.2.4] - 2026-04-24

### Changed
- Smoke-test release via the new repo's GitHub Actions publish workflow (crates.io + npm).
  `tiktoken` path-dep bumped to 3.1.4. No code changes.
- npm package is now built on CI via `wasm-pack --scope goliapkg`, so the
  shipped `package.json` reflects wasm-pack 0.14 defaults (`files` list no
  longer includes the legacy `tiktoken_wasm_bg.js`).

## [3.2.3] - 2026-04-24

### Changed
- Migrated from `goliajp/airs` mono-repo to standalone `goliajp/rust-tiktoken` (shares repo with `tiktoken`).
  No code changes; `repository` URL updated, `tiktoken` path-dep bumped to 3.1.3. WASM binary
  is bit-identical to the 3.2.2 release.

## [3.2.2] - 2026-03-07

- Previous release (from `goliajp/airs`).
