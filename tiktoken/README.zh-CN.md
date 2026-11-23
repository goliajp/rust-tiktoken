# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#许可证)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square&logo=rust)](Cargo.toml)
[![Downloads](https://img.shields.io/crates/d/tiktoken?style=flat-square)](https://crates.io/crates/tiktoken)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

最快的 Rust BPE 分词器 — 原生比 tiktoken-rs 快 22〜83 倍（中日文散文 50〜82 倍），比 fastokens 快 1.5〜12 倍，浏览器内（wasm）比 gpt-tokenizer 快 5.8〜7.0 倍。手写扫描器同时覆盖 ASCII 与 CJK，词表按 key 长度分层，重复片段整片记忆。兼容 OpenAI [tiktoken](https://github.com/openai/tiktoken)，并支持**所有主流 LLM 分词器** — OpenAI、Llama 3、DeepSeek、Qwen、Mistral、Kimi、GLM、MiniMax。

## 特性

- **多厂商**：20 种编码，覆盖 8 家厂商（OpenAI、Meta、DeepSeek、阿里巴巴、Mistral、Moonshot、智谱、MiniMax）
- **高性能**：每套编码都有查表实现的预分词器（运行时不用正则），cl100k 与 o200k 系列用 SIMD 判定片段起点，词表按 key 长度分层，重复片段整片记忆且跨线程共享，混合 BPE 合并
- **并行编码**：32 KiB 及以上的文本自动用多个线程编码，线程池内置、只依赖 std
- **费用估算**：覆盖 11 家厂商共 153 个模型
- **体积紧凑**：20 份词表共 7.8 MB 编译期嵌入，且可按词表退订 —— 只要 cl100k 的构建仅带 373 KB 数据，cargo 也只下载启用的厂商的数据
- **零分配计数**：`count()` 不分配 token 向量

## 性能

计时前先断言各实现的 token 输出完全一致；每个数字是一次完整处理，预热后取
9 轮中位数。语料在所有基准衣架中逐字节相同（`bench-compare/`、
`benches/bench_python.py`、`web/bench/`）。

#### 原生 — Apple M4 Pro Mac mini、单线程、`encode`

`TIKTOKEN_THREADS=1 cargo run --release -p bench-compare`

| 语料 | Python tiktoken 0.14 | tiktoken-rs 0.12 | **tiktoken** | vs rs | vs Python |
|---|---|---|---|---|---|
| 短文本 (13 B) | 1.3 µs | 602 ns | **28 ns** | **22x** | **46x** |
| 中等文本 (900 B) | 23.2 µs | 19.8 µs | **431 ns** | **46x** | **54x** |
| 英文文本 (45 KB) | 1,100 µs | 955 µs | **19.9 µs** | **48x** | **55x** |
| 中文散文 (4.3 KB) | 114 µs | 97.8 µs | **2.0 µs** | **50x** | **58x** |
| 日文散文 (4.6 KB) | 131 µs | 119 µs | **1.4 µs** | **82x** | **91x** |
| 多语混排 ×50 (4.5 KB) | 122 µs | 108 µs | **2.5 µs** | **43x** | **49x** |
| 对抗语料：CJK 无重复 (3.9 KB) | 114 µs | 94.0 µs | **2.6 µs** | **36x** | **43x** |
| 代码 (3.9 KB) | 179 µs | 163 µs | **1.9 µs** | **83x** | **92x** |

o200k_base 单线程比 tiktoken-rs 快 12〜80 倍。`count()` 在短文本上比 `encode`
最多快 36%，长文本上两者相当 —— 它从不分配 id 向量。用默认线程数时，45 KB
文本只要 6.2 µs。

#### 与最快的其他实现对比 — Apple M4 Pro Mac mini、单线程、cl100k_base、`encode`

我们找到的、有可能更快的实现，都在同一个进程里用同样的输入测，先逐个 id
核对一致；这套测试程序独立于 `bench-compare`。数字是 3 遍的中位数，
`tiktoken` 用的是 crates.io 上的 5.2.0。

| 语料 | **tiktoken** | [fastokens](https://crates.io/crates/fastokens) 0.3.6 | [HF tokenizers](https://crates.io/crates/tokenizers) 1.0-rc.5 | [toks](https://github.com/actual-computer/toks) 0.3.3 | [toktok-rs](https://crates.io/crates/toktok-rs) 0.5 | [magi-tokenizer](https://crates.io/crates/magi-tokenizer) 0.1 | [bpe-openai](https://crates.io/crates/bpe-openai) 0.3 |
|---|---|---|---|---|---|---|---|
| 短文本 (13 B) | **25 ns** | 61 ns | 112 ns | 42 ns | 32 ns | 27 ns | 147 ns |
| 中等文本 (900 B) | **431 ns** | 750 ns | 733 ns | 606 ns | 1.3 µs | 1.0 µs | 8.2 µs |
| 英文文本 (45 KB) | **19.9 µs** | 35.6 µs | 32.3 µs | 29.9 µs | 61.2 µs | 49.5 µs | 405 µs |
| 中文散文 (4.3 KB) | **2.0 µs** | 8.0 µs | 4.3 µs | 12.9 µs | 10.1 µs | 12.2 µs | 37.7 µs |
| 日文散文 (4.6 KB) | **1.4 µs** | 17.2 µs | 4.1 µs | 18.8 µs | 15.8 µs | 22.8 µs | 41.1 µs |
| 多语混排 ×50 (4.5 KB) | **2.6 µs** | 3.9 µs | 5.8 µs | 17.8 µs | 12.7 µs | 18.9 µs | 40.0 µs |
| 对抗语料：CJK 无重复 (3.9 KB) | **2.7 µs** | 5.3 µs | 5.4 µs | 17.2 µs | 15.3 µs | 19.4 µs | 56.1 µs |
| 代码 (3.9 KB) | **1.9 µs** | 3.9 µs | 3.4 µs | 2.8 µs | 6.1 µs | 6.6 µs | 46.1 µs |
| 英文书籍 (200 KB) | **111 µs** | 184 µs | 205 µs | 144 µs | 341 µs | 715 µs | 2,854 µs |
| Rust 源码 (100 KB) | **56.1 µs** | 95.8 µs | 97.6 µs | 79.8 µs | 181 µs | 263 µs | 1,412 µs |

- **fastokens**：单线程比本库慢 1.5〜12 倍。默认线程数下，英文书籍它用 50.3 µs、
  Rust 源码 36.3 µs，本库是 16.7 µs 和 12.5 µs（分别快 3.0 倍和 2.9 倍）
- **短文本**：13 字节那一行，magi-tokenizer 与本库相差 4〜12%
- **首次遇到的文本**（缓存清空、单线程）：比 fastokens 快 1.7〜6.2 倍，英文书籍 228 µs
  对 392 µs；13 字节以上的文本比其他库至少快 1.17 倍；13 字节时几个库都落在计时器的
  42 ns 分辨率上，算平手。默认线程数下的英文书籍，fastokens 的工作线程保留缓存时，
  本库领先 8〜20%
- **toks** 列出的是关闭整篇文本记忆的成绩。打开后，再次编码*同一篇*文本时它直接返回
  存下的结果，比本库重新编码快 3〜20 倍
- 其余试过的库 —— tiktoken-rs、tokensim、riptoken、kitoken、basetenkenizer、
  Hugging Face tokenizers 0.23 —— 每一行都至少慢 3.4 倍（Hugging Face 0.23 和
  kitoken 在 Rust 源码上输出的 id 不一致，那一行没有计时）。Gigatoken 只能从 Python
  调用，含调用开销慢 1.4〜11 倍

#### 浏览器内 — Apple M4 Pro Mac mini、Chromium

在 `web/` 目录 `npm run bench` —— 本 crate 编译为 wasm，对比另一个 wasm 分词器和两个
主流 JavaScript 分词器。各列都是 `encode`。

| 语料 | tiktoken-fast 0.1 (wasm) | gpt-tokenizer 4.1 | js-tiktoken 1.0 | **tiktoken (wasm)** |
|---|---|---|---|---|
| 中文散文 (4.3 KB) | 6.4 µs | 31.9 µs | 6,625 µs | **4.8 µs** |
| 日文散文 (4.6 KB) | 6.6 µs | 29.2 µs | 12,825 µs | **4.2 µs** |
| 多语混排 ×50 (4.5 KB) | 7.0 µs | 40.3 µs | 3,975 µs | **6.3 µs** |
| 对抗语料：CJK 无重复 (3.9 KB) | 6.7 µs | 42.2 µs | 2,756 µs | **6.1 µs** |
| 英文文本 (45 KB) | 118 µs | 277 µs | 9,638 µs | **47.7 µs** |
| 代码 (3.9 KB) | 8.9 µs | 30.4 µs | 1,120 µs | **5.1 µs** |

两份多语混排语料在 tiktoken-fast 见过之后，差距缩小到 10% 左右；首次遇到时它慢
1.8〜6.6 倍。在 Safari 里，本库比 gpt-tokenizer 快 3.3〜7.9 倍，比 tiktoken-fast
快 1.5〜2.2 倍。

对抗语料不含任何重复片段，各实现的记忆化全部失效，领先依然成立。

<details>
<summary>为什么更快？</summary>

| | tiktoken | tiktoken-rs | Python tiktoken |
|---|---|---|---|
| 预分词 | 每套编码都用查表实现的扫描器，Unicode 表按切分规则所依据的 Unicode 版本生成；cl100k 与 o200k 系列用 SIMD（NEON / SSE2 / wasm simd128）一次判定 64 字节内的片段起点 | 总是走正则 | 总是走正则 |
| 正则引擎 | 运行时不用（`regex` 只是开发依赖，作为测试扫描器的对照实现） | `fancy-regex`（回溯） | `regex` 经 PyO3 + FFI 开销 |
| 词表查找 | 按 key 长度分层：15 字节以内的 key 压缩存进一张开放寻址表（每个缓存行两个槽），更长的用带哈希 tag 的 arena 槽，1〜2 字节另有直查表 | `HashMap<Vec<u8>>`（20 万次分配） | PyO3 背后的 Rust `HashMap` |
| 重复片段 | 不在词表里的片段，合并结果写进共享的查找表，所有线程都能复用；长片段另有线程局部和共享的缓存 | 无 | 无 |
| BPE 合并 | 混合：栈上线性扫描（短片段）+ 堆（长片段）；查不到的短片段按 token id 成批合并 | O(n*m) 线性扫描 | O(n*m) 线性扫描 |
| 零分配 `count()` | 有 | 无 | 无 |

扫描器由属性测试（每轮数十万条随机输入）和短字符串穷举钉死在 `regex` crate 的语义上；每套编码由
53,275 组差分 fixture 钉死在厂商自己的分词器上。

基准测试源码：[`benches/`](benches/)、[`../bench-compare/`](../bench-compare/)。
可通过 `cargo bench` / `cargo run -p bench-compare` 复现。

</details>

## 安装

```toml
[dependencies]
tiktoken = "5"
```

运行时依赖只有 `ruzstd` 和启用的词表数据 crate。早期版本的 `parallel` 特性仍然保留，写了它的清单照样能构建，但它不再有任何作用：不开它，长文本也会并行编码。

### 选择词表

20 种编码默认全开（7.8 MB 词表数据）。关掉默认特性、只点名用到的，二进制里就只带这些：

```toml
# 只要 GPT-4o / GPT-5 —— 词表数据 815 KB 而非 7.8 MB
tiktoken = { version = "5", default-features = false, features = ["vocab-o200k_base"] }

# 或者按厂商整组
tiktoken = { version = "5", default-features = false, features = ["vocab-openai"] }
```

实测 `examples/count_tokens`（release，Linux x86_64）：全词表 9,463,280 字节，仅 `vocab-cl100k_base` 2,550,256 字节。

词表数据按厂商分在各自的 crate 里（`tiktoken-vocab-openai`、`-meta`、`-deepseek`、`-qwen`、`-mistral`、`-moonshot`、`-zhipu`、`-minimax`），由上面的特性引入，所以 cargo 只下载你启用的厂商：只用 OpenAI 约 1.5 MB，全部是 8.0 MB。

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
| `o200k_base` | OpenAI | `vocab-o200k_base` | 815 KB | GPT-4o, GPT-4.1, GPT-4.5, GPT-5–5.6 (incl. Sol/Terra/Luna), GPT-6.x（推定，官方未公布）, o1, o3, o4-mini |
| `o200k_harmony` | OpenAI | `vocab-o200k_base` | — | gpt-oss（harmony 对话格式） |
| `cl100k_base` | OpenAI | `vocab-cl100k_base` | 373 KB | GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, text-embedding-*, davinci-002, babbage-002 |
| `p50k_base` | OpenAI | `vocab-p50k_base` | 55 B + 基座 | text-davinci-002/003, code-davinci-*, code-cushman-* |
| `p50k_edit` | OpenAI | `vocab-p50k_base` | — | text-davinci-edit-*, code-davinci-edit-* |
| `r50k_base` | OpenAI | `vocab-r50k_base` | 182 KB | GPT-3 时代：davinci, curie, babbage, ada |
| `gpt2` | OpenAI | `vocab-r50k_base` | — | GPT-2（`r50k_base` 的别名） |
| `llama3` | Meta | `vocab-llama3` | 111 KB + 基座 | Llama 3, 3.1, 3.2, 3.3 |
| `llama4` | Meta | `vocab-llama4` | 816 KB | Llama 4 Scout / Maverick, Muse |
| `deepseek_v3` | DeepSeek | `vocab-deepseek_v3` | 514 KB | DeepSeek V3, R1 |
| `deepseek_v4` | DeepSeek | `vocab-deepseek_v3` | — | DeepSeek V4 Pro / Flash, V4.1 Flash（V3 词表 + V4 特殊 token） |
| `qwen2` | 阿里巴巴 | `vocab-qwen2` | 564 KB | Qwen 2.5, Qwen 3 |
| `qwen3_5` | 阿里巴巴 | `vocab-qwen3_5` | 989 KB | Qwen 3.5 – 3.8, Qwen 4（推定，官方未公布） |
| `mistral_v3` | Mistral | `vocab-mistral_v3` | 525 KB | Mistral, Mixtral, Large 3, Small 4（Tekken 分词器） |
| `mistral_v15` | Mistral | `vocab-mistral_v15` | 853 KB | Mistral Large 4 (Tekken v15) |
| `kimi_k2` | Moonshot | `vocab-kimi_k2` | 659 KB | Kimi K2 / K2.5 / K2.6 |
| `kimi_k3` | Moonshot | `vocab-kimi_k2` | — | Kimi K3 / K3.1（K2 词表 + K3 特殊 token） |
| `glm4` | 智谱 | `vocab-glm4` | 578 KB | GLM-4.5 / 4.6 / 4.7 |
| `glm5` | 智谱 | `vocab-glm5` | 6 KB + 基座 | GLM-5 / 5.1 / 5.2 / 5.3 |
| `minimax_m2` | MiniMax | `vocab-minimax_m2` | 822 KB | MiniMax M2 / M2.1 / M2.5 / M2.7 / M3 / M3.1 |

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

`encode`、`count`、`encode_with_special_tokens` 和 `count_with_special_tokens`
遇到 32 KiB 及以上的文本时，会把它切成若干块，交给进程内共用的工作线程池编码；线程池在第一次这样调用时创建。输出与单线程编码完全一致。

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let text = "word ".repeat(20_000);
let tokens = enc.encode(&text); // 100 KB：多线程编码

// 每次调用可用的线程数，含调用方线程；1 = 单线程，0 = 恢复默认
tiktoken::set_parallelism(1);
assert_eq!(enc.encode(&text), tokens);
tiktoken::set_parallelism(0);
```

- 默认线程数是性能核的数量（系统不区分核类型时为全部核数）；环境变量 `TIKTOKEN_THREADS` 是正整数时以它为准。
- `TIKTOKEN_THREADS=1` 或 `set_parallelism(1)` 关闭并行：每次调用都在调用方线程上执行，不会启动任何工作线程。
- 同一时间只有一个并行调用。发现线程池正忙的调用（比如应用本身已经在多个线程上编码）不会等待，直接在自己的线程上执行，所以进程里用于编码的线程不会超过设定的数量。
- `encode_parallel` 现在等同于 `encode`，只为兼容旧代码保留，不再需要 `parallel` 特性。
- `wasm32` 上没有线程，所有调用都是单线程。

### 费用估算

```rust
use tiktoken::pricing;

// 快速估算
let cost = pricing::estimate_cost("gpt-4o", 1_000_000, 500_000).unwrap();

// 带 prompt 缓存
let model = pricing::get_model("claude-opus-5.5").unwrap();
let cost = model.estimate_cost_with_cache(500_000, 500_000, 200_000);

// 缓存写入按缓存保留时长计价，这里是 1 小时；高速档和后台完成窗口各有估算函数
assert_eq!(model.cache_write_per_1m(60), Some(8.0));
let fast = pricing::get_model("gpt-6-sol").unwrap().estimate_priority_cost(1_000, 1_000);
let background = pricing::get_model("gpt-6-luna").unwrap().estimate_background_cost(1_000, 1_000, 8);

// 按厂商列出所有模型
let models = pricing::models_by_provider(pricing::Provider::DeepSeek);

// 各家 API 实际使用的 id 写法也能解析 —— 本表用点、API 用短横线,带发布日期后缀,
// 以及 Bedrock / Vertex 的平台装饰
let r = pricing::resolve_model("us.anthropic.claude-opus-5").unwrap();
assert_eq!(r.model.id, "claude-opus-5");
assert!(matches!(r.matched, pricing::Match::Normalized { .. }));

// `estimate_cost` 走同一套解析;`get_model` 保持精确匹配
assert!(pricing::estimate_cost("claude-haiku-4-5-20251001", 1_000, 1_000).is_some());
```

支持 OpenAI、Anthropic、Google、Meta、DeepSeek、阿里巴巴、Mistral、Moonshot、
智谱、MiniMax、Voyage 共 153 个模型。

## WebAssembly

npm 包 [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)，可直接在浏览器和 Node.js 中使用。包用 WebAssembly SIMD 构建，需要 Chrome 91、Firefox 89 或 Safari 16.4 及以上。

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
wasm-pack build --target web --release --scope goliapkg
```

</details>

<!-- ECOSYSTEM BEGIN (generated — edit ecosystem.toml, not this block) -->

## 生态系统

**tiktoken** · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · [chunkedrs](https://crates.io/crates/chunkedrs) · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## 许可证

代码采用 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE) 双重许可，由你任选其一。

内嵌的词表不在此列，各自沿用发布方的许可证：[LICENSE-3RD-PARTY](LICENSE-3RD-PARTY) 列出每套词表的来源和许可证，[`licenses/`](licenses/) 收录全文，[NOTICE](NOTICE) 收录这些许可证要求的署名。其中一些许可证对使用附有条件：Llama 许可证（月活超过 7 亿的产品要向 Meta 另行申请许可）、`mistral_v15` 的 Mistral 许可证（经营「Model as a Service」业务的要另签商业许可）、Kimi 和 MiniMax 在规模超过门槛时的署名展示条款，以及 DeepSeek 的使用限制条款。不启用某套词表的 `vocab-*` 特性，构建产物里就不含这套词表。

Built with Llama.
