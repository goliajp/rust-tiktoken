# Changelog

All notable changes to this crate / npm package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
