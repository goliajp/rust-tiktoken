# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#许可证)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square&logo=rust)](Cargo.toml)
[![Downloads](https://img.shields.io/crates/d/tiktoken?style=flat-square)](https://crates.io/crates/tiktoken)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

最快的 Rust BPE 分词器 — 原生比 tiktoken-rs 快 5〜49 倍（中日文散文 15〜17 倍），浏览器内（wasm）比 gpt-tokenizer 快 2〜4 倍。手写扫描器同时覆盖 ASCII 与 CJK，词表按 key 长度分层，重复片段整片记忆。兼容 OpenAI [tiktoken](https://github.com/openai/tiktoken)，并支持**所有主流 LLM 分词器** — OpenAI、Llama 3、DeepSeek、Qwen、Mistral、Kimi、GLM、MiniMax。

## 特性

- **多厂商**：17 种编码，覆盖 8 家厂商（OpenAI、Meta、DeepSeek、阿里巴巴、Mistral、Moonshot、智谱、MiniMax）
- **高性能**：手写扫描器覆盖 ASCII 与 CJK（绕开正则）、词表按 key 长度分层、重复片段整片记忆、混合 BPE 合并
- **并行编码**：可选的 rayon 多线程编码，适用于长文本
- **费用估算**：覆盖 10 家厂商共 107 个模型
- **体积紧凑**：17 份词表共 5.1 MB 编译期嵌入，且可按词表退订 —— 只要 cl100k 的构建仅带 373 KB 数据
- **零分配计数**：`count()` 不分配 token 向量

## 性能

计时前先断言各实现的 token 输出完全一致；每个数字是一次完整处理，预热后取
9 轮中位数。语料在所有基准衣架中逐字节相同（`bench-compare/`、
`benches/bench_python.py`、`web/bench/`）。

#### 原生 — Apple M4 Mac mini、单线程、`encode`

`cargo run --release -p bench-compare`

| 语料 | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken** | vs rs | vs Python |
|---|---|---|---|---|---|
| 短文本 (13 B) | 1.6 µs | 1,081 ns | **33 ns** | **33x** | **48x** |
| 中等文本 (900 B) | 31.9 µs | 52.2 µs | **1.1 µs** | **47x** | **29x** |
| 英文文本 (45 KB) | 1,500 µs | 2,498 µs | **51.5 µs** | **49x** | **29x** |
| 中文散文 (4.3 KB) | 119.8 µs | 134.7 µs | **8.1 µs** | **17x** | **15x** |
| 日文散文 (4.6 KB) | 131.0 µs | 144.6 µs | **8.6 µs** | **17x** | **15x** |
| 多语混排 ×50 (4.5 KB) | 138.9 µs | 160.3 µs | **15.2 µs** | **11x** | **9.2x** |
| 对抗语料：CJK 无重复 (3.9 KB) | 131.7 µs | 141.2 µs | **25.9 µs** | **5.5x** | **5.1x** |
| 代码 (3.9 KB) | 263.7 µs | 317.7 µs | **11.1 µs** | **29x** | **24x** |

o200k_base 比例一致（对 tiktoken-rs 5〜48 倍）。`count()` 比 `encode`
再快 5〜15% —— 它从不分配 id 向量。

#### 浏览器内 — Mac Studio (M4 Max)、Chromium

在 `web/` 目录 `npm run bench` —— 本 crate 编译为 wasm，对比两个主流
JavaScript 分词器。

| 语料 | gpt-tokenizer 3.4 | js-tiktoken 1.0 | **tiktoken (wasm)** |
|---|---|---|---|
| 中文散文 (4.3 KB) | 36.8 µs | 8,029 µs | **13.4 µs** |
| 日文散文 (4.6 KB) | 27.4 µs | 15,862 µs | **13.5 µs** |
| 多语混排 ×50 (4.5 KB) | 41.2 µs | 4,665 µs | **24.2 µs** |
| 对抗语料：CJK 无重复 (3.9 KB) | 49.6 µs | 3,832 µs | **40.3 µs** |
| 英文文本 (45 KB) | 478 µs | 7,010 µs | **112.5 µs** |
| 代码 (3.9 KB) | 76.0 µs | 916 µs | **19.5 µs** |

对抗语料不含任何重复片段，各实现的记忆化全部失效 —— 那是下界，领先在下界处
依然成立。

<details>
<summary>为什么更快？</summary>

| | tiktoken | tiktoken-rs | Python tiktoken |
|---|---|---|---|
| 预分词 | 手写扫描器同时覆盖 ASCII 与 CJK（汉字/假名/谚文、全角形式）；正则只作判准与罕见形状的兜底 | 总是走正则 | 总是走正则 |
| 正则引擎（兜底） | `regex`（DFA，线性时间） | `fancy-regex`（回溯） | `regex` 经 PyO3 + FFI 开销 |
| 词表查找 | 按 key 长度分层：1〜2 字节直查表、3〜8 字节内联槽开放寻址、更长的带 tag 的 arena 槽 | `HashMap<Vec<u8>>`（20 万次分配） | PyO3 背后的 Rust `HashMap` |
| 重复片段 | 线程局部直接映射记忆化，字节键 | 无 | 无 |
| BPE 合并 | 混合：栈上线性扫描（短片段）+ 堆（长片段） | O(n*m) 线性扫描 | O(n*m) 线性扫描 |
| 零分配 `count()` | 有 | 无 | 无 |

扫描器由属性测试钉死在正则语义上（每轮数十万条随机输入）；每套编码由
44,518 组差分 fixture 钉死在厂商自己的分词器上。

基准测试源码：[`benches/`](benches/)、[`../bench-compare/`](../bench-compare/)。
可通过 `cargo bench` / `cargo run -p bench-compare` 复现。

</details>

## 安装

```toml
[dependencies]
tiktoken = "4"

# 可选：大文本多线程编码
tiktoken = { version = "4", features = ["parallel"] }
```

### 选择词表

17 种编码默认全开（5.1 MB 词表数据）。关掉默认特性、只点名用到的，二进制里就只带这些：

```toml
# 只要 GPT-4o / GPT-5 —— 词表数据 815 KB 而非 5.1 MB
tiktoken = { version = "4", default-features = false, features = ["vocab-o200k_base"] }

# 或者按厂商整组
tiktoken = { version = "4", default-features = false, features = ["vocab-openai"] }
```

实测 `examples/count_tokens`（release）：全词表 6,480,912 字节，仅 `vocab-cl100k_base` 2,226,704 字节。

没编进来的编码就是不存在：`list_encodings()` 不列出它，`get_encoding()` 返回 `None`。`pricing` 表与词表无关，所以只做费用估算时可以一个 `vocab-*` 都不开。

厂商组：`vocab-openai`、`vocab-meta`、`vocab-deepseek`、`vocab-qwen`、`vocab-mistral`、`vocab-moonshot`、`vocab-zhipu`、`vocab-minimax`，以及默认的 `vocabs-all`。逐词表的 feature 见下表。

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

`数据`列是该编码的 feature 给二进制增加的词表字节。共用同一份数据文件的编码第二次不再计费；标 `+ 基座` 的三个是秩对齐扩展，只存自己的尾部。

| 编码 | 厂商 | Feature | 数据 | 适用模型 |
|---|---|---|---|---|
| `o200k_base` | OpenAI | `vocab-o200k_base` | 815 KB | GPT-4o, GPT-4.1, GPT-4.5, GPT-5–5.6 (incl. Sol/Terra/Luna), o1, o3, o4-mini |
| `o200k_harmony` | OpenAI | `vocab-o200k_base` | — | gpt-oss（harmony 对话格式） |
| `cl100k_base` | OpenAI | `vocab-cl100k_base` | 373 KB | GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, text-embedding-*, davinci-002, babbage-002 |
| `p50k_base` | OpenAI | `vocab-p50k_base` | 55 B + 基座 | text-davinci-002/003, code-davinci-*, code-cushman-* |
| `p50k_edit` | OpenAI | `vocab-p50k_base` | — | text-davinci-edit-*, code-davinci-edit-* |
| `r50k_base` | OpenAI | `vocab-r50k_base` | 182 KB | GPT-3 时代：davinci, curie, babbage, ada |
| `gpt2` | OpenAI | `vocab-r50k_base` | — | GPT-2（`r50k_base` 的别名） |
| `llama3` | Meta | `vocab-llama3` | 111 KB + 基座 | Llama 3, 3.1, 3.2, 3.3, 4 |
| `deepseek_v3` | DeepSeek | `vocab-deepseek_v3` | 514 KB | DeepSeek V3, R1 |
| `deepseek_v4` | DeepSeek | `vocab-deepseek_v3` | — | DeepSeek V4 Pro / Flash（V3 词表 + V4 特殊 token） |
| `qwen2` | 阿里巴巴 | `vocab-qwen2` | 564 KB | Qwen 2.5, Qwen 3 |
| `mistral_v3` | Mistral | `vocab-mistral_v3` | 525 KB | Mistral, Mixtral（Tekken 分词器） |
| `kimi_k2` | Moonshot | `vocab-kimi_k2` | 659 KB | Kimi K2 / K2.5 / K2.6 |
| `kimi_k3` | Moonshot | `vocab-kimi_k2` | — | Kimi K3（K2 词表 + K3 特殊 token） |
| `glm4` | 智谱 | `vocab-glm4` | 578 KB | GLM-4.5 / 4.6 / 4.7 |
| `glm5` | 智谱 | `vocab-glm5` | 6 KB + 基座 | GLM-5 / 5.2 |
| `minimax_m2` | MiniMax | `vocab-minimax_m2` | 822 KB | MiniMax M2 / M2.1 / M2.5 / M2.7 |

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

支持 OpenAI、Anthropic、Google、Meta、DeepSeek、阿里巴巴、Mistral 共 107 个模型。

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

<!-- ECOSYSTEM BEGIN (generated — edit ecosystem.toml, not this block) -->

## 生态系统

**tiktoken** · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · [chunkedrs](https://crates.io/crates/chunkedrs) · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## 许可证

采用 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE) 双重许可，由你任选其一。

第三方词表数据许可证：参见 [LICENSE-3RD-PARTY](LICENSE-3RD-PARTY)。
