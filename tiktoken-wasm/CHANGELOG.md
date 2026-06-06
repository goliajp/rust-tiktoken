# Changelog

All notable changes to this crate / npm package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
