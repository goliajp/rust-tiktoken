# Changelog

All notable changes to this crate / npm package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Dual-licensed under **MIT OR Apache-2.0** (previously MIT only):
  `LICENSE-MIT` + `LICENSE-APACHE`, `license = "MIT OR Apache-2.0"`.

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
