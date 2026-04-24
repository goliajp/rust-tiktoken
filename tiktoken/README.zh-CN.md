# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square&logo=rust)](Cargo.toml)
[![Downloads](https://img.shields.io/crates/d/tiktoken?style=flat-square)](https://crates.io/crates/tiktoken)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

最快的 Rust BPE 分词器 — 比 tiktoken-rs 快 7-10 倍。兼容 OpenAI [tiktoken](https://github.com/openai/tiktoken)，并支持**所有主流 LLM 分词器** — OpenAI、Llama 3、DeepSeek、Qwen 和 Mistral。

## 特性

- **多厂商**：9 种编码，覆盖 5 家厂商（OpenAI、Meta、DeepSeek、阿里巴巴、Mistral）
- **高性能**：Arena 词表存储、堆加速 BPE 合并、DFA 正则
- **并行编码**：可选的 rayon 多线程编码，适用于长文本
- **费用估算**：覆盖 7 家厂商共 57 个模型
- **体积紧凑**：ruzstd 压缩词表数据，编译期嵌入
- **零分配计数**：`count()` 不分配 token 向量

## 性能

所有基准测试在 Apple M4 Mac mini 上单线程运行。三个实现的 token 输出已验证完全一致。

#### cl100k_base encode

| 输入 | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken 3.1** | vs tiktoken-rs | vs Python |
|---|---|---|---|---|---|
| 短文本 (13 B) | 1,700 ns | 1,248 ns | **118 ns** | **10.6x** | **14x** |
| 中等文本 (900 B) | 32.2 us | 53.8 us | **7.2 us** | **7.5x** | **4.5x** |
| 长文本 (45 KB) | 1,500 us | 2,611 us | **366 us** | **7.1x** | **4.1x** |
| Unicode (4.5 KB) | 141 us | 164 us | **101 us** | **1.6x** | **1.4x** |
| 代码 (3.9 KB) | 247 us | 264 us | **42 us** | **6.3x** | **5.9x** |

#### o200k_base encode

| 输入 | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken 3.1** | vs tiktoken-rs | vs Python |
|---|---|---|---|---|---|
| 短文本 (13 B) | 1,600 ns | 1,051 ns | **115 ns** | **9.1x** | **14x** |
| 中等文本 (900 B) | 58.3 us | 56.2 us | **7.1 us** | **7.9x** | **8.2x** |
| 长文本 (45 KB) | 2,900 us | 2,799 us | **365 us** | **7.7x** | **7.9x** |
| Unicode (4.5 KB) | 204 us | 187 us | **99 us** | **1.9x** | **2.1x** |
| 代码 (3.9 KB) | 332 us | 253 us | **41 us** | **6.2x** | **8.1x** |

<details>
<summary>为什么更快？</summary>

| | tiktoken | tiktoken-rs | Python tiktoken |
|---|---|---|---|
| 词表存储 | Arena（单次分配，缓存友好） | `HashMap<Vec<u8>>`（20 万次分配） | PyO3 背后的 Rust `HashMap` |
| 正则引擎 | `regex`（DFA，线性时间） | `fancy-regex`（回溯） | `regex` 经 PyO3 + FFI 开销 |
| 哈希表 | 自定义开放寻址 + `FxHash` | `rustc-hash` v1 | 标准 `HashMap` |
| BPE 合并 | 堆加速 O(n log n) | O(n*m) 线性扫描 | O(n*m) 线性扫描 |
| 零分配 `count()` | 有 | 无 | 无 |

基准测试源码：[`benches/`](benches/)。可通过 `cargo bench` 复现。

</details>

## 安装

```toml
[dependencies]
tiktoken = "3"

# 可选：大文本多线程编码
tiktoken = { version = "3", features = ["parallel"] }
```

## 快速开始

```rust
// 按编码名称
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode("hello world");
let text = enc.decode_to_string(&tokens).unwrap();
assert_eq!(text, "hello world");

// 按模型名称 — 支持所有厂商
let enc = tiktoken::encoding_for_model("gpt-4o").unwrap();
let count = enc.count("hello world"); // 零分配快速路径

let enc = tiktoken::encoding_for_model("llama-3.1-70b").unwrap();
let enc = tiktoken::encoding_for_model("deepseek-v3").unwrap();
let enc = tiktoken::encoding_for_model("qwen2.5-72b").unwrap();
```

## 支持的编码

| 编码 | 厂商 | 适用模型 |
|---|---|---|
| `o200k_base` | OpenAI | GPT-4o, GPT-4o-mini, o1, o3, o4-mini |
| `cl100k_base` | OpenAI | GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, text-embedding-* |
| `p50k_base` | OpenAI | text-davinci-002/003, code-davinci-*, code-cushman-* |
| `p50k_edit` | OpenAI | text-davinci-edit-*, code-davinci-edit-* |
| `r50k_base` | OpenAI | GPT-3 时代：davinci, curie, babbage, ada |
| `llama3` | Meta | Llama 3, 3.1, 3.2, 3.3, 4 |
| `deepseek_v3` | DeepSeek | DeepSeek V3, R1 |
| `qwen2` | 阿里巴巴 | Qwen 2.5, Qwen 3 |
| `mistral_v3` | Mistral | Mistral, Mixtral（Tekken 分词器） |

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

// 带特殊 token 识别的计数
let count = enc.count_with_special_tokens("hello<|endoftext|>world");
```

### 并行编码

```rust
// 需要 `parallel` feature
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode_parallel("...非常长的文本...");
// 输出完全一致，>= 4KB 文本时使用 rayon 并行
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
let models = pricing::models_by_provider(pricing::Provider::DeepSeek);
```

支持 OpenAI、Anthropic、Google、Meta、DeepSeek、阿里巴巴、Mistral 共 57 个模型。

## WebAssembly

npm 包 [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)，可直接在浏览器和 Node.js 中使用。

```bash
npm install @goliapkg/tiktoken-wasm
```

```typescript
import init, { getEncoding, encodingForModel, estimateCost } from '@goliapkg/tiktoken-wasm'

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

<!-- ECOSYSTEM BEGIN (synced by claws/opensource/scripts/sync-ecosystem.py — edit ecosystem.toml, not this block) -->

## 生态系统

**tiktoken** · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · [chunkedrs](https://crates.io/crates/chunkedrs) · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## 许可证

[MIT](LICENSE)

第三方词表数据许可证：参见 [LICENSE-3RD-PARTY](LICENSE-3RD-PARTY)。
