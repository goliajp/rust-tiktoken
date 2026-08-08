# rust-tiktoken

[![tiktoken on crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust&label=tiktoken)](https://crates.io/crates/tiktoken)
[![tiktoken-wasm on npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm?style=flat-square&logo=npm&label=tiktoken-wasm)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-tiktoken/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-tiktoken/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#license)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) · **[tiktoken.golia.jp](https://tiktoken.golia.jp)** — live in-browser playground

The fastest Rust BPE tokenizer, plus its WebAssembly bindings. Drop-in compatible with OpenAI [tiktoken](https://github.com/openai/tiktoken) and the mainstream open models (Llama 3, DeepSeek, Qwen, Mistral). A hand-written ASCII fast-path makes ASCII text **15–40x faster than tiktoken-rs** (≈2x on CJK/Unicode).

## Crates in this workspace

| Path | Crate / Package | Description | Version |
|:-----|:----------------|:------------|:--------|
| [`tiktoken/`](tiktoken/) | [`tiktoken`](https://crates.io/crates/tiktoken) | Rust BPE tokenizer — 17 encodings, 107 models, multi-provider pricing | [![crates.io](https://img.shields.io/crates/v/tiktoken.svg?style=flat-square)](https://crates.io/crates/tiktoken) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`tiktoken-wasm`](https://crates.io/crates/tiktoken-wasm) (Rust) | WASM binding crate for the above | [![crates.io](https://img.shields.io/crates/v/tiktoken-wasm.svg?style=flat-square)](https://crates.io/crates/tiktoken-wasm) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) (npm) | Same, published to npm via `wasm-pack` | [![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm.svg?style=flat-square)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) |

> The two crates live in one workspace and are **versioned in lockstep** — every release bumps and publishes both at the same version number.

## Highlights

- **ASCII fast-path pre-tokenizer** — resolves the common ASCII pieces (letters, digits, punctuation, contractions) without invoking the regex engine; **2.3–5.5x faster** `encode` / `count` on ASCII text for cl100k / o200k / qwen2 / deepseek. Unicode/CJK transparently falls back to the regex.
- **17 encodings · 107 models · 10 providers** — OpenAI (GPT-4/4o/4.1/4.5, GPT-5.x, o1/o3/o4-mini, gpt-oss), Llama 3/4, DeepSeek V3/V4, Qwen, Mistral, Kimi K2/K3, GLM-4/5, MiniMax M2; plus USD cost estimation (Anthropic & Google included for pricing).
- **Lean & portable** — arena-based vocabulary, hybrid linear/heap BPE merge, optional rayon parallelism, allocation-free `count()`, pure Rust with zero C dependencies, a tiny wasm build, and zstd-compressed vocab embedded at compile time.

Full API, supported-model tables, and benchmarks live in the per-crate READMEs: **[`tiktoken/`](tiktoken/README.md)** · **[`tiktoken-wasm/`](tiktoken-wasm/README.md)**.

## Quick start

### Rust

```toml
[dependencies]
tiktoken = "3.5"
```

```rust
// by encoding name
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode("hello world");
assert_eq!(enc.decode_to_string(&tokens).unwrap(), "hello world");

// count without allocating a token vector
let n = enc.count("The quick brown fox.");

// or resolve by model name
let enc = tiktoken::encoding_for_model("gpt-4o").unwrap();
```

### WebAssembly (browser / Node.js)

```bash
npm install @goliapkg/tiktoken-wasm
```

```js
import init, { getEncoding } from '@goliapkg/tiktoken-wasm'
await init()
const enc = getEncoding('o200k_base')
const tokens = enc.encode('hello world')
```

## Performance

On an Apple M4 Mac mini, ASCII `encode` / `count` is **15–40x faster than tiktoken-rs** and **~20–40x faster than Python tiktoken** (≈2x on CJK/Unicode, which defers to the regex). Full per-encoding tables and methodology: [`tiktoken/README.md#performance`](tiktoken/README.md#performance).

## Build

```bash
cargo test -p tiktoken
cargo fmt --all --check
cargo clippy --workspace --lib -- -D warnings

# WASM (requires wasm-pack: cargo install wasm-pack)
cd tiktoken-wasm
wasm-pack build --target web --release --scope goliapkg
```

## Release

`tiktoken` and `tiktoken-wasm` are versioned in lockstep and released together via git-flow (no PRs):

```bash
git flow release start X.Y.Z
# bump versions to X.Y.Z: tiktoken/Cargo.toml, tiktoken-wasm/Cargo.toml
# (and its tiktoken path-dep); finalize both CHANGELOGs
git flow release finish X.Y.Z                       # merge → master, tag vX.Y.Z, back-merge develop
git tag -a tiktoken-wasm-vX.Y.Z vX.Y.Z^{commit} -m "tiktoken-wasm X.Y.Z"
git push origin master develop vX.Y.Z tiktoken-wasm-vX.Y.Z
# tag `v*` publishes the tiktoken crate; `tiktoken-wasm-v*` publishes the wasm crate + npm
```

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
