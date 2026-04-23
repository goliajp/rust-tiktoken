# rust-tiktoken

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

OpenAI の tiktoken BPE トークナイザーの高速な純 Rust 実装と、ブラウザ / Node.js
向けの WebAssembly バインディング。

## このワークスペースの crate

| パス | Crate / Package | 説明 | バージョン |
|:-----|:----------------|:-----|:-----------|
| [`tiktoken/`](tiktoken/) | [`tiktoken`](https://crates.io/crates/tiktoken) | Rust BPE トークナイザー — 9 エンコーディング、57 モデル、各社料金 | [![crates.io](https://img.shields.io/crates/v/tiktoken.svg?style=flat-square)](https://crates.io/crates/tiktoken) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`tiktoken-wasm`](https://crates.io/crates/tiktoken-wasm) (Rust) | 上記の WASM バインディング crate | [![crates.io](https://img.shields.io/crates/v/tiktoken-wasm.svg?style=flat-square)](https://crates.io/crates/tiktoken-wasm) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) (npm) | 同じものを `wasm-pack` で npm へ公開 | [![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm.svg?style=flat-square)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) |

両 crate が同一リポジトリにあるのは、`tiktoken` のリリースごとに
`tiktoken-wasm` も連動リリースが必要だからです。単一ワークスペースで
同期するほうが、2 リポジトリ間の調整より安価です。

## ビルド

```bash
# Rust
cargo test -p tiktoken
cargo clippy --workspace --all-targets

# WASM (wasm-pack が必要: cargo install wasm-pack)
cd tiktoken-wasm
wasm-pack build --target web --release --scope goliapkg
```

## リリース

各 crate は独立したリリースサイクルを持ち、タグ接頭辞で分岐します:

```bash
# tiktoken (Rust crate)
git flow release start 3.1.4
# ... tiktoken/Cargo.toml と tiktoken/CHANGELOG.md を更新 ...
git flow release finish 3.1.4      # v3.1.4 タグを作成、CI が公開

# tiktoken-wasm (Rust crate + npm パッケージ)
# ... tiktoken-wasm/Cargo.toml と pkg/package.json のバージョンを更新 ...
git tag -a tiktoken-wasm-v3.2.4 -m "release tiktoken-wasm 3.2.4"
git push origin tiktoken-wasm-v3.2.4   # CI が crates.io と npm の両方へ公開
```

## ライセンス

[MIT](LICENSE)
