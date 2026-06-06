# Changelog

## [3.1.5] - 2026-06-06

### Added
- `model_to_encoding` now supports `gpt-5` and `gpt-4.5` prefixes → `o200k_base`.
- `model_to_encoding` strips a leading `ft:` prefix so fine-tuned model IDs (e.g.
  `ft:gpt-4o:my-org::abc123`) resolve to the base model's encoding.
- Azure-style `gpt-35-turbo` alias → `cl100k_base`.

### Fixed
- `davinci-002` and `babbage-002` now correctly route to `cl100k_base` (they were
  greedily caught by the `davinci`/`babbage` prefix in the `r50k_base` block).

### Changed
- Pricing refreshed against official 2026-06 docs across all 7 providers:
  - 13 price corrections (notably OpenAI `o1-mini` to match `o3-mini` at $1.10/$4.40,
    Anthropic `claude-opus-4`/`claude-sonnet-4` cache values to cache-read convention,
    Alibaba Qwen 2.5 family now uses official Model Studio split rates, Mistral
    `mistral-large` cut to $0.5/$1.5, `mistral-medium` repositioned to $1.5/$7.5).
  - 21 `DEPRECATED` doc-comments (OpenAI shutdown 2026-10-23, Anthropic Claude 3.x
    retired, claude-opus-4/sonnet-4 retire 2026-06-15, Gemini 2.0 Flash shutdown,
    Gemini 1.5 family removed from official pricing, DeepSeek `deepseek-chat`/
    `deepseek-reasoner` deprecate 2026-07-24, Mistral `pixtral-large`).
  - Module-level caveats document the Anthropic cache-read convention, the Meta
    llama legacy-Together source, and the `gemini-2.5-pro` single-tier limitation.
- Dropped `rust-version` MSRV pin from workspace; tracks current stable.
- CI release workflow: Node 20 → 24, `wasm-pack` install switched to
  `taiki-e/install-action` (prebuilt binary, ~5s vs ~5–10 min compile).

## [3.1.4] - 2026-04-24

### Changed
- Smoke-test release via the new repo's GitHub Actions publish workflow.
  No code changes.

## [3.1.3] - 2026-04-24

### Changed
- Migrated from `goliajp/airs` mono-repo to standalone `goliajp/rust-tiktoken` (shares repo with `tiktoken-wasm`).
  No code changes; `repository` URL updated.

## [3.0.1] - 2026-03-07

### Changed

- Updated benchmark tables in README to v3.0 numbers (all three languages)

## [3.0.0] - 2026-03-07

### Added

- **Multi-provider tokenizer support**: Llama 3 (`llama3`), DeepSeek V3 (`deepseek_v3`), Qwen 2 (`qwen2`), Mistral V3 (`mistral_v3`) encodings
- **Parallel encoding**: `encode_parallel()` method behind `parallel` feature flag, uses rayon for texts >= 4KB
- **count_with_special_tokens()**: counting method that recognizes special tokens, matching `encode_with_special_tokens()` behavior
- **Multi-provider pricing**: Meta Llama, DeepSeek, Alibaba Qwen, Mistral models added to pricing module (39 total models across 7 providers)
- HuggingFace oracle tests: cross-validation against Python HF tokenizer output for all new encodings
- Property-based testing: 125,000 random input roundtrip tests via proptest
- Third-party license documentation (`LICENSE-3RD-PARTY`)
- Vocab conversion scripts (`scripts/convert_vocab.py`, `scripts/generate_hf_oracle.py`)

### Changed

- **Breaking**: internal architecture rewritten for performance
  - `FxHashMap<Vec<u8>, u32>` replaced with arena-based `Vocab` (single allocation, cache-friendly)
  - BPE merge algorithm: O(n*m) linear scan replaced with O(n log n) heap-accelerated merge (BinaryHeap + doubly-linked list)
  - Pre-tokenization abstracted behind `PreTokenizer` trait (`RegexPreTokenizer` implementation)
- Vocabulary data now zstd-compressed at rest (~63% compression ratio, 7MB -> 2.6MB for OpenAI vocabs)
- Internal modules (`vocab`, `merge`, `pretokenize`) use `pub(crate)` visibility
- `model_to_encoding()` now supports Llama, DeepSeek, Qwen, and Mistral model name prefixes

### Fixed

- NBSP overflow bug in whitespace lookahead emulation: single multi-byte whitespace characters (e.g. U+00A0) no longer cause empty piece underflow
- DeepSeek ZWJ character handling: format characters (Unicode Cf category) no longer skipped by regex

### Performance

- Arena-based vocabulary: single contiguous allocation replaces 200k individual `Vec<u8>` heap allocations
- Heap-accelerated BPE merge: O(n log n) vs O(n*m) for the ~5% of pieces that enter the merge path
- Fast paths for 1-byte and 2-byte pieces in BPE merge

## [2.1.1] - 2026-03-07

### Changed

- Upgraded Rust edition 2021 -> 2024, MSRV 1.85 -> 1.94
- Upgraded `criterion` dev-dependency 0.5 -> 0.8
- Migrated Cargo.toml metadata to workspace inheritance (edition, rust-version, license, repository, homepage, authors)
- Moved to `airs` monorepo workspace

## [2.1.0] - 2026-03-06

### Performance

- Rank-cached BPE merge: only recomputes 2 neighbor ranks per merge step
- 2-byte piece fast path in `byte_pair_merge`
- Unicode encode/count improved ~52%, ASCII text improved 2-10%

### Added

- `decode_to_string()` method for ergonomic UTF-8 decoding
- `model_to_encoding()` as public API
- `o200k_base` encoding (GPT-4o, o1, o3, o4-mini)
- Multi-provider pricing module (`tiktoken::pricing`) -- OpenAI, Anthropic Claude, Google Gemini (26 models)
- WebAssembly bindings (`tiktoken-wasm`) with encode/decode/count/pricing
- React demo app (`examples/react-app`)
- Criterion benchmarks with Python tiktoken 0.12.0 comparison
- Trilingual documentation (English, simplified Chinese, Japanese)
- 93 tests (82 unit + 11 doc tests), 97% line coverage

### Changed

- **Breaking**: rewritten from scratch -- new engine, new dependencies, new API surface
- Replaced `pcre2` (backtracking) with `regex` crate (DFA engine)
- Replaced `HashMap` with `FxHashMap` for faster small-key hashing
- Replaced `lazy_static` with `OnceLock` for encoding instance caching
- Upgraded `base64` 0.21 -> 0.22
- Upgraded edition 2021 -> 2024, MSRV set to 1.85
- Removed dependencies: `anyhow`, `maplit`, `rust_decimal`, `rust_decimal_macros`

### Removed

- `src/models.rs` and `src/price.rs` (replaced by `src/pricing.rs`)

## [1.0.1] - 2023-08-18

- Initial release on crates.io
