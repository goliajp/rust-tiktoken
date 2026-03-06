# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen?style=flat-square)](src/)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-blue?style=flat-square&logo=rust)](Cargo.toml)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

最快的 Rust BPE 分词器，兼容 OpenAI [tiktoken](https://github.com/openai/tiktoken)。**比 tiktoken-rs 快 7–10 倍**，**比 Python tiktoken 快 4–15 倍**。

## 性能

所有基准测试在 Apple M4 Mac mini 上单线程运行。三个实现的 token 输出已验证完全一致。

#### cl100k_base encode

| 输入 | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken 2.1** | vs tiktoken-rs | vs Python |
|---|---|---|---|---|---|
| 短文本 (13 B) | 1,700 ns | 1,248 ns | **118 ns** | **10.5x** | **14x** |
| 中等文本 (900 B) | 32.2 µs | 53.8 µs | **7.3 µs** | **7.3x** | **4.4x** |
| 长文本 (45 KB) | 1,500 µs | 2,611 µs | **373 µs** | **7.0x** | **4.0x** |
| Unicode (4.5 KB) | 141 µs | 164 µs | **97 µs** | **1.7x** | **1.5x** |
| 代码 (3.9 KB) | 247 µs | 264 µs | **34 µs** | **7.7x** | **7.3x** |

#### o200k_base encode

| 输入 | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken 2.1** | vs tiktoken-rs | vs Python |
|---|---|---|---|---|---|
| 短文本 (13 B) | 1,600 ns | 1,051 ns | **116 ns** | **9.1x** | **14x** |
| 中等文本 (900 B) | 58.3 µs | 56.2 µs | **7.3 µs** | **7.7x** | **8.0x** |
| 长文本 (45 KB) | 2,900 µs | 2,799 µs | **374 µs** | **7.5x** | **7.8x** |
| Unicode (4.5 KB) | 204 µs | 187 µs | **104 µs** | **1.8x** | **2.0x** |
| 代码 (3.9 KB) | 332 µs | 253 µs | **33 µs** | **7.6x** | **10x** |

<details>
<summary>为什么更快？</summary>

| | tiktoken | tiktoken-rs | Python tiktoken |
|---|---|---|---|
| 正则引擎 | `regex`（DFA，线性时间） | `fancy-regex`（回溯） | `regex` 经 PyO3 + FFI 开销 |
| 哈希表 | `FxHashMap`（小 key 更快） | `rustc-hash` v1 | PyO3 背后的 Rust `HashMap` |
| BPE 合并 | rank 缓存（仅重算 2 邻居） | 标准实现 | 标准实现 |
| 零分配 `count()` | 有 | 无 | 无 |
| 运行时依赖 | 4 crates | 16 crates | — |

基准测试源码：[`benches/`](benches/)。可通过 `cargo bench` 复现。

</details>

## 安装

```toml
[dependencies]
tiktoken = "2.1"
```

## 快速开始

```rust
// 按编码名称
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode("hello world");
let text = enc.decode_to_string(&tokens).unwrap();
assert_eq!(text, "hello world");

// 按模型名称
let enc = tiktoken::encoding_for_model("gpt-4o").unwrap();
let count = enc.count("hello world"); // 零分配快速路径
```

## 支持的编码

| 编码 | 适用模型 |
|---|---|
| `o200k_base` | GPT-4o, GPT-4o-mini, o1, o3, o4-mini |
| `cl100k_base` | GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, text-embedding-ada-002, text-embedding-3-* |
| `p50k_base` | text-davinci-002/003, code-davinci-*, code-cushman-* |
| `p50k_edit` | text-davinci-edit-*, code-davinci-edit-* |
| `r50k_base` | text-davinci-001, text-curie, text-babbage, text-ada, davinci, curie, babbage, ada |

## API

### 编解码

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();

let tokens = enc.encode("hello world");           // Vec<u32>
let text = enc.decode_to_string(&tokens).unwrap(); // String
let bytes = enc.decode(&tokens);                   // Vec<u8>
```

### 特殊 token

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode_with_special_tokens("hello<|endoftext|>world");
// tokens 中会包含 <|endoftext|> 对应的特殊 token id
```

### Token 计数

```rust
let enc = tiktoken::get_encoding("o200k_base").unwrap();
let count = enc.count("敏捷的棕色狐狸跳过了懒狗。");
// 比 encode().len() 更快 — 不分配 token 向量
```

### 费用估算

```rust
use tiktoken::pricing;

// 快速估算
let cost = pricing::estimate_cost("gpt-4o", 1_000_000, 500_000).unwrap();

// 带 prompt 缓存
let model = pricing::get_model("claude-opus-4").unwrap();
let cost = model.estimate_cost_with_cache(500_000, 500_000, 200_000);

// 按厂商列出所有模型
let models = pricing::models_by_provider(pricing::Provider::Google);
```

支持 OpenAI、Anthropic Claude、Google Gemini 共 26 个模型。

## WebAssembly

npm 包 [`@goliapkg/tokenrs-wasm`](https://www.npmjs.com/package/@goliapkg/tokenrs-wasm)，可直接在浏览器和 Node.js 中使用。

```bash
npm install @goliapkg/tokenrs-wasm
```

```typescript
import init, { getEncoding, encodingForModel, estimateCost } from '@goliapkg/tokenrs-wasm'

await init()

const enc = getEncoding('cl100k_base')
const tokens = enc.encode('hello world')    // Uint32Array
const text = enc.decode(tokens)             // "hello world"
const count = enc.count('hello world')      // 2

enc.free() // 使用完毕释放 WASM 内存
```

| 打包工具 | 插件 |
|---------|--------|
| Vite | [vite-plugin-wasm](https://www.npmjs.com/package/vite-plugin-wasm) + [vite-plugin-top-level-await](https://www.npmjs.com/package/vite-plugin-top-level-await) |
| webpack 5 | 内置 `asyncWebAssembly` 实验特性 |
| Next.js | [next.config.js `webpack.experiments`](https://nextjs.org/docs/app/api-reference/next-config-js/webpack) |

完整示例参见 [`examples/react-app`](examples/react-app/)。

<details>
<summary>从源码构建</summary>

```bash
# 需要 wasm-pack: cargo install wasm-pack
cd tiktoken-wasm
wasm-pack build --target web --release
```

</details>

## 许可证

[MIT](LICENSE)
