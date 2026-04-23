# rust-tiktoken

[English](README.md) | **简体中文** | [日本語](README.ja.md)

高性能纯 Rust 版 OpenAI tiktoken BPE 分词器，以及用于浏览器 / Node.js 的 WebAssembly 绑定。

## 本 workspace 的 crate

| 路径 | Crate / Package | 说明 | 版本 |
|:-----|:----------------|:-----|:-----|
| [`tiktoken/`](tiktoken/) | [`tiktoken`](https://crates.io/crates/tiktoken) | Rust BPE 分词器 —— 9 套编码、57 个模型、多厂商价格 | [![crates.io](https://img.shields.io/crates/v/tiktoken.svg?style=flat-square)](https://crates.io/crates/tiktoken) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`tiktoken-wasm`](https://crates.io/crates/tiktoken-wasm) (Rust) | 上述分词器的 WASM 绑定 crate | [![crates.io](https://img.shields.io/crates/v/tiktoken-wasm.svg?style=flat-square)](https://crates.io/crates/tiktoken-wasm) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) (npm) | 同上，通过 `wasm-pack` 发布到 npm | [![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm.svg?style=flat-square)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) |

这两个 crate 同仓，是因为每次 `tiktoken` 发版都要联动发一版 `tiktoken-wasm`；
放一个 workspace 比跨两仓做协调便宜得多。

## 构建

```bash
# Rust
cargo test -p tiktoken
cargo clippy --workspace --all-targets

# WASM（需要 wasm-pack：cargo install wasm-pack）
cd tiktoken-wasm
wasm-pack build --target web --release --scope goliapkg
```

## 发布

两个 crate 各自独立发版，按 tag 前缀分派：

```bash
# tiktoken（Rust crate）
git flow release start 3.1.4
# … 更新 tiktoken/Cargo.toml 和 tiktoken/CHANGELOG.md …
git flow release finish 3.1.4      # 打 v3.1.4 tag，CI 自动发布

# tiktoken-wasm（Rust crate + npm 包）
# … 更新 tiktoken-wasm/Cargo.toml 和 pkg/package.json 版本 …
git tag -a tiktoken-wasm-v3.2.4 -m "release tiktoken-wasm 3.2.4"
git push origin tiktoken-wasm-v3.2.4   # CI 同时发 crates.io + npm
```

## 许可证

[MIT](LICENSE)
