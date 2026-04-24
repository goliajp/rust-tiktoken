# rust-tiktoken

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

High-performance pure-Rust implementation of OpenAI's tiktoken BPE tokenizer,
plus WebAssembly bindings for browser / Node.js consumption.

## Crates in this workspace

| Path | Crate / Package | Description | Version |
|:-----|:----------------|:------------|:--------|
| [`tiktoken/`](tiktoken/) | [`tiktoken`](https://crates.io/crates/tiktoken) | Rust BPE tokenizer — 9 encodings, 57 models, multi-provider pricing | [![crates.io](https://img.shields.io/crates/v/tiktoken.svg?style=flat-square)](https://crates.io/crates/tiktoken) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`tiktoken-wasm`](https://crates.io/crates/tiktoken-wasm) (Rust) | WASM binding crate for the above | [![crates.io](https://img.shields.io/crates/v/tiktoken-wasm.svg?style=flat-square)](https://crates.io/crates/tiktoken-wasm) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) (npm) | Same, published to npm via `wasm-pack` | [![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm.svg?style=flat-square)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) |

The two crates live in one repo because every `tiktoken` release requires a
coordinated `tiktoken-wasm` publish; keeping them in sync in a single workspace
is cheaper than coordinating two repos.

## Build

```bash
# Rust
cargo test -p tiktoken
cargo clippy --workspace --all-targets

# WASM (requires wasm-pack: cargo install wasm-pack)
cd tiktoken-wasm
wasm-pack build --target web --release --scope goliapkg
```

## Release

Each crate has an independent release cycle, selected by tag prefix:

```bash
# tiktoken (Rust crate)
git flow release start 3.1.4
# ... bump tiktoken/Cargo.toml + tiktoken/CHANGELOG.md ...
git flow release finish 3.1.4      # creates v3.1.4 tag, CI publishes

# tiktoken-wasm (Rust crate + npm package)
# ... bump tiktoken-wasm/Cargo.toml + pkg/package.json version ...
git tag -a tiktoken-wasm-v3.2.4 -m "release tiktoken-wasm 3.2.4"
git push origin tiktoken-wasm-v3.2.4   # CI publishes both crates.io + npm
```

## License

[MIT](LICENSE)
