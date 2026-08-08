# rust-tiktoken

[![tiktoken on crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust&label=tiktoken)](https://crates.io/crates/tiktoken)
[![tiktoken-wasm on npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm?style=flat-square&logo=npm&label=tiktoken-wasm)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-tiktoken/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-tiktoken/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#许可证)

[English](README.md) | **简体中文** | [日本語](README.ja.md) · **[tiktoken.golia.jp](https://tiktoken.golia.jp)** —— 浏览器在线体验

最快的 Rust BPE 分词器，以及它的 WebAssembly 绑定。兼容 OpenAI [tiktoken](https://github.com/openai/tiktoken)，并支持主流开源模型（Llama 3、DeepSeek、Qwen、Mistral、Kimi、GLM、MiniMax）。手写扫描器同时覆盖 ASCII 与 CJK，配合按 key 长度分层的词表与整片记忆，**原生比 tiktoken-rs 快 5〜49 倍**，**浏览器内比 gpt-tokenizer 快 2〜4 倍** —— 中日文散文同样领先。

## 本 workspace 的 crate

| 路径 | Crate / Package | 说明 | 版本 |
|:-----|:----------------|:-----|:-----|
| [`tiktoken/`](tiktoken/) | [`tiktoken`](https://crates.io/crates/tiktoken) | Rust BPE 分词器 —— 17 套编码、107 个模型、多厂商价格 | [![crates.io](https://img.shields.io/crates/v/tiktoken.svg?style=flat-square)](https://crates.io/crates/tiktoken) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`tiktoken-wasm`](https://crates.io/crates/tiktoken-wasm) (Rust) | 上述分词器的 WASM 绑定 crate | [![crates.io](https://img.shields.io/crates/v/tiktoken-wasm.svg?style=flat-square)](https://crates.io/crates/tiktoken-wasm) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) (npm) | 同上，通过 `wasm-pack` 发布到 npm | [![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm.svg?style=flat-square)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) |

> 两个 crate 同属一个 workspace，**版本号始终同步（lockstep）**—— 每次发版都把两者一起升到同一版本号并一起发布。

## 亮点

- **手写预分词器，覆盖 ASCII 与 CJK** —— 字母、数字、标点、缩写、汉字/假名/谚文与全角形式都不经过正则引擎直接解析；正则仍是判准（属性测试钉死等价），也是罕见形状的兜底。
- **17 套编码 · 107 个模型 · 10 家厂商** —— OpenAI（GPT-4/4o/4.1/4.5、GPT-5.x、o1/o3/o4-mini、gpt-oss）、Llama 3/4、DeepSeek V3/V4、Qwen、Mistral、Kimi K2/K3、GLM-4/5、MiniMax M2；并提供美元成本估算（价格含 Anthropic、Google）。
- **轻量可移植** —— 按 key 长度分层的词表与整片记忆、线性/堆混合 BPE 合并、可选 rayon 并行、零分配 `count()`、零 C 依赖的纯 Rust、自足的 wasm 构建、zstd 压缩词表编译期内嵌。

完整 API、支持模型表、基准测试见各 crate 的 README：**[`tiktoken/`](tiktoken/README.md)** · **[`tiktoken-wasm/`](tiktoken-wasm/README.md)**。

## 快速开始

### Rust

```toml
[dependencies]
tiktoken = "3.8"
```

```rust
// 按编码名获取
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode("hello world");
assert_eq!(enc.decode_to_string(&tokens).unwrap(), "hello world");

// 不分配 token 向量的计数
let n = enc.count("The quick brown fox.");

// 或按模型名解析
let enc = tiktoken::encoding_for_model("gpt-4o").unwrap();
```

### WebAssembly（浏览器 / Node.js）

```bash
npm install @goliapkg/tiktoken-wasm
```

```js
import init, { getEncoding } from '@goliapkg/tiktoken-wasm'
await init()
const enc = getEncoding('o200k_base')
const tokens = enc.encode('hello world')
```

## 性能

在 Apple M4 Mac mini 上，`encode` **比 tiktoken-rs 快 5〜49 倍**、**比 Python tiktoken 快 5〜29 倍**：ASCII 29〜49 倍，中日文散文 15〜17 倍，对抗性无重复 CJK 语料 5 倍。浏览器内（wasm）比 gpt-tokenizer 快 2〜4 倍。完整对比表与方法见 [`tiktoken/README.md#performance`](tiktoken/README.md#performance)。

## 构建

```bash
cargo test -p tiktoken
cargo fmt --all --check
cargo clippy --workspace --lib -- -D warnings

# WASM（需要 wasm-pack：cargo install wasm-pack）
cd tiktoken-wasm
wasm-pack build --target web --release --scope goliapkg
```

## 发布

`tiktoken` 与 `tiktoken-wasm` 版本同步，通过 git-flow 一起发布（不走 PR）：

```bash
git flow release start X.Y.Z
# 把版本升到 X.Y.Z：tiktoken/Cargo.toml、tiktoken-wasm/Cargo.toml
#（及其 tiktoken path 依赖）；确定两个 CHANGELOG。
git flow release finish X.Y.Z                       # 合并到 master、打 vX.Y.Z tag、回合 develop
git tag -a tiktoken-wasm-vX.Y.Z vX.Y.Z^{commit} -m "tiktoken-wasm X.Y.Z"
git push origin master develop vX.Y.Z tiktoken-wasm-vX.Y.Z
# tag `v*` 发布 tiktoken crate；`tiktoken-wasm-v*` 发布 wasm crate + npm
```

## 许可证

采用 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE) 双重许可，由你任选其一。
