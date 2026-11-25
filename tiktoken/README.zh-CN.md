# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#许可证)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square&logo=rust)](Cargo.toml)
[![Downloads](https://img.shields.io/crates/d/tiktoken?style=flat-square)](https://crates.io/crates/tiktoken)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

tiktoken 是速度最快的 Rust BPE 分词器。原生编码速度是 tiktoken-rs 的 22〜83 倍（中日文文章为 50〜82 倍）、fastokens 的 1.5〜12 倍。在浏览器中，wasm 版本的编码速度是 gpt-tokenizer 的 5.8〜7.0 倍。分词器用手写扫描器处理 ASCII 和 CJK，按键长组织词表，并缓存整个片段的编码结果供重复使用。本库兼容 OpenAI [tiktoken](https://github.com/openai/tiktoken)，支持所有主流大语言模型的分词器，包括 OpenAI、Llama 3、DeepSeek、Qwen、Mistral、Kimi、GLM 和 MiniMax。

## 特性

- **多厂商**：8 家厂商的 20 种编码（OpenAI、Meta、DeepSeek、阿里巴巴、Mistral、Moonshot、智谱、MiniMax）
- **快**：每种编码的预分词都靠查表完成，运行时不用正则；cl100k 和 o200k 系列用 SIMD 找片段起点；词表按 key 的长度分层；整个片段的编码结果在线程间共享；BPE 合并用混合算法
- **并行编码**：32 KiB 及以上的文本自动分给多个线程编码，用的是内置线程池，只依赖标准库
- **费用估算**：覆盖 11 家厂商的 153 个模型
- **体积小**：20 份词表一共 7.8 MB，编译时内嵌，每份都能单独去掉。只要 cl100k 的构建只带 373 KB 数据，cargo 也只下载你启用的厂商的数据
- **计数不分配内存**：`count()` 不构造 token 向量

## 性能

测试程序在计时前逐一核对各实现输出的 token，确认完全一致后才开始计时。每轮处理一次完整语料，表中耗时取预热后 9 轮测量的中位数。各测试程序（`bench-compare/`、`benches/bench_python.py`、`web/bench/`）使用的语料逐字节相同。

#### 原生单线程 `encode` 性能（Apple M4 Pro Mac mini）

`TIKTOKEN_THREADS=1 cargo run --release -p bench-compare`

| 语料 | Python tiktoken 0.14 | tiktoken-rs 0.12 | **tiktoken** | 相对 tiktoken-rs 的速度倍数 | 相对 Python 的速度倍数 |
|---|---|---|---|---|---|
| 短文本（13 B） | 1.3 µs | 602 ns | **28 ns** | **22x** | **46x** |
| 中等文本（900 B） | 23.2 µs | 19.8 µs | **431 ns** | **46x** | **54x** |
| 英文文章（45 KB） | 1,100 µs | 955 µs | **19.9 µs** | **48x** | **55x** |
| 中文文章（4.3 KB） | 114 µs | 97.8 µs | **2.0 µs** | **50x** | **58x** |
| 日文文章（4.6 KB） | 131 µs | 119 µs | **1.4 µs** | **82x** | **91x** |
| CJK 混排 ×50（4.5 KB） | 122 µs | 108 µs | **2.5 µs** | **43x** | **49x** |
| 无重复片段的 CJK 对抗语料（3.9 KB） | 114 µs | 94.0 µs | **2.6 µs** | **36x** | **43x** |
| 代码（3.9 KB） | 179 µs | 163 µs | **1.9 µs** | **83x** | **92x** |

使用 o200k_base 时，本库的单线程速度是 tiktoken-rs 的 12〜80 倍。`count()` 无需分配 ID 向量，处理短文本时比 `encode` 最多快 36%，处理长文本时两者速度相近。使用默认线程数时，编码 45 KB 的文本耗时 6.2 µs。

#### 各高性能实现的单线程 `encode` 耗时（Apple M4 Pro Mac mini，cl100k_base）

我们将找到的所有可能更快的实现纳入比较，使用独立于 `bench-compare` 的测试程序。在同一进程中，各实现处理相同的输入，计时前同样逐一核对 token ID。表中耗时取 3 次测量的中位数，`tiktoken` 使用 crates.io 发布的 5.2.0 版。

| 语料 | **tiktoken** | [fastokens](https://crates.io/crates/fastokens) 0.3.6 | [HF tokenizers](https://crates.io/crates/tokenizers) 1.0-rc.5 | [toks](https://github.com/actual-computer/toks) 0.3.3 | [toktok-rs](https://crates.io/crates/toktok-rs) 0.5 | [magi-tokenizer](https://crates.io/crates/magi-tokenizer) 0.1 | [bpe-openai](https://crates.io/crates/bpe-openai) 0.3 |
|---|---|---|---|---|---|---|---|
| 短文本（13 B） | **25 ns** | 61 ns | 112 ns | 42 ns | 32 ns | 27 ns | 147 ns |
| 中等文本（900 B） | **431 ns** | 750 ns | 733 ns | 606 ns | 1.3 µs | 1.0 µs | 8.2 µs |
| 英文文章（45 KB） | **19.9 µs** | 35.6 µs | 32.3 µs | 29.9 µs | 61.2 µs | 49.5 µs | 405 µs |
| 中文文章（4.3 KB） | **2.0 µs** | 8.0 µs | 4.3 µs | 12.9 µs | 10.1 µs | 12.2 µs | 37.7 µs |
| 日文文章（4.6 KB） | **1.4 µs** | 17.2 µs | 4.1 µs | 18.8 µs | 15.8 µs | 22.8 µs | 41.1 µs |
| CJK 混排 ×50（4.5 KB） | **2.6 µs** | 3.9 µs | 5.8 µs | 17.8 µs | 12.7 µs | 18.9 µs | 40.0 µs |
| 无重复片段的 CJK 对抗语料（3.9 KB） | **2.7 µs** | 5.3 µs | 5.4 µs | 17.2 µs | 15.3 µs | 19.4 µs | 56.1 µs |
| 代码（3.9 KB） | **1.9 µs** | 3.9 µs | 3.4 µs | 2.8 µs | 6.1 µs | 6.6 µs | 46.1 µs |
| 英文书籍（200 KB） | **111 µs** | 184 µs | 205 µs | 144 µs | 341 µs | 715 µs | 2,854 µs |
| Rust 源码（100 KB） | **56.1 µs** | 95.8 µs | 97.6 µs | 79.8 µs | 181 µs | 263 µs | 1,412 µs |

- 本库的单线程速度是 fastokens 的 1.5〜12 倍。使用默认线程数时，fastokens 编码英文书籍耗时 50.3 µs，编码 Rust 源码耗时 36.3 µs。本库分别耗时 16.7 µs 和 12.5 µs，速度分别是 fastokens 的 3.0 倍和 2.9 倍
- 编码 13 字节的字符串时，magi-tokenizer 与本库的耗时相差 4%〜12%
- 清空缓存后首次以单线程编码时，本库的速度是 fastokens 的 1.7〜6.2 倍（英文书籍分别耗时 228 µs 和 392 µs）。输入超过 13 字节时，本库的速度至少是其余各库的 1.17 倍。输入恰好为 13 字节时，有几个库的耗时都等于计时器的最小刻度（42 ns）。使用默认线程数编码英文书籍时，如果 fastokens 的 worker 保留了缓存，本库仍快 8%〜20%
- 表中 toks 的成绩使用了关闭全文缓存的配置。启用缓存后，toks 再次处理相同文本时会直接返回已有结果，速度是本库重新编码的 3〜20 倍。本库的 [`CoreBpe::memo`](#重复文本) 也支持全文缓存。在这组语料上使用 cl100k 和 o200k 时，直接返回缓存的 ID，速度是 toks 全文缓存的 1.3〜6 倍。如果和 toks 一样将 ID 复制到缓冲区，本库处理 45 KB 文本时比 toks 慢 10%，处理 13 字节文本时速度是 toks 的 5 倍，其他结果介于两者之间
- 测试还涵盖 tiktoken-rs、tokensim、riptoken、kitoken、basetenkenizer 和 Hugging Face tokenizers 0.23。在每项有计时结果的测试中，本库的速度至少是这些库的 3.4 倍。Hugging Face 0.23 和 kitoken 处理 Rust 源码时输出的 ID 不一致，因此这项测试没有记录两者的耗时。Gigatoken 只能从 Python 调用，计入调用开销后，本库的速度是它的 1.4〜11 倍

#### 浏览器性能（Apple M4 Pro Mac mini，Chromium）

在 `web/` 目录运行 `npm run bench`，即可比较本 crate 的 wasm 版本、另一款 wasm 分词器和两款主流 JavaScript 分词器。表中各列均为 `encode` 的耗时。

| 语料 | tiktoken-fast 0.1 (wasm) | gpt-tokenizer 4.1 | js-tiktoken 1.0 | **tiktoken (wasm)** |
|---|---|---|---|---|
| 中文文章（4.3 KB） | 6.4 µs | 31.9 µs | 6,625 µs | **4.8 µs** |
| 日文文章（4.6 KB） | 6.6 µs | 29.2 µs | 12,825 µs | **4.2 µs** |
| CJK 混排 ×50（4.5 KB） | 7.0 µs | 40.3 µs | 3,975 µs | **6.3 µs** |
| 无重复片段的 CJK 对抗语料（3.9 KB） | 6.7 µs | 42.2 µs | 2,756 µs | **6.1 µs** |
| 英文文章（45 KB） | 118 µs | 277 µs | 9,638 µs | **47.7 µs** |
| 代码（3.9 KB） | 8.9 µs | 30.4 µs | 1,120 µs | **5.1 µs** |

处理两组混合文字语料时，tiktoken-fast 如果已经编码过相同的输入，与本库的速度差距就在约 10% 以内。首次处理这些语料时，本库的速度是它的 1.8〜6.6 倍。在 Safari 中，本库的速度是 gpt-tokenizer 的 3.3〜7.9 倍、tiktoken-fast 的 1.5〜2.2 倍。

对抗语料中的每个片段都只出现一次，各实现都无法复用缓存的结果。本库在这组语料上同样更快。

<details>
<summary>为什么更快？</summary>

| | tiktoken | tiktoken-rs | Python tiktoken |
|---|---|---|---|
| 预分词 | 所有编码都用扫描器查 Unicode 表，生成表时采用各切分规则对应的 Unicode 版本。cl100k 和 o200k 系列用 SIMD（NEON、SSE2 或 AVX2、wasm simd128）每次检查 64 字节，找出片段起点 | 每次都执行正则表达式 | 每次都执行正则表达式 |
| 正则引擎 | 运行时无需正则引擎（`regex` 仅作开发依赖，用于在测试中核对扫描结果） | `fancy-regex`（支持回溯） | 通过 PyO3 调用 `regex`，有 FFI 开销 |
| 词表查找 | 按键长分层。最多 15 字节的键紧凑存放在同一张开放寻址表中，每个缓存行有两个槽位。更长的键存入带哈希标记的 arena 槽位。1〜2 字节的键可直接按索引查表 | `HashMap<Vec<u8>>`（20 万次分配） | 通过 PyO3 访问 Rust `HashMap` |
| 重复片段 | 分词器将词表以外的片段合并后，把结果写入共享查找表，供所有线程复用。长片段另有线程局部缓存和共享缓存 | 无 | 无 |
| BPE 合并 | 短片段在栈上线性扫描，长片段用堆处理。查表未命中的短片段按 token ID 成批合并 | O(n*m) 线性扫描 | O(n*m) 线性扫描 |
| 零分配 `count()` | 有 | 无 | 无 |

属性测试每次运行都生成数十万条随机输入，配合短字符串穷举测试，核对扫描器与 `regex` crate 的输出是否一致。另有 53,275 组差分测试用例，逐种编码核对本库与厂商分词器的输出。

基准测试源码见 [`benches/`](benches/) 和 [`../bench-compare/`](../bench-compare/)。运行 `cargo bench` 或 `cargo run -p bench-compare` 可复现测试。

</details>

## 安装

```toml
[dependencies]
tiktoken = "5"
```

运行时只依赖 `ruzstd` 和已启用的词表数据 crate。为兼容旧的依赖声明，本库保留了早期版本的 `parallel` feature。该 feature 已无实际作用，长文本会自动并行编码。

### 选择词表

默认启用全部 20 种编码，词表数据共占 7.8 MB。关闭默认 feature 后，可按需启用编码，这样二进制文件只包含选定的词表。

```toml
# 只要 GPT-4o / GPT-5 —— 词表数据 815 KB 而非 7.8 MB
tiktoken = { version = "5", default-features = false, features = ["vocab-o200k_base"] }

# 或者按厂商整组
tiktoken = { version = "5", default-features = false, features = ["vocab-openai"] }
```

在 Linux x86_64 上以 release 模式构建 `examples/count_tokens`，包含全部词表的可执行文件为 9,463,280 字节，只启用 `vocab-cl100k_base` 时为 2,550,256 字节。

每家厂商的词表数据单独存放在一个 crate 中（`tiktoken-vocab-openai`、`-meta`、`-deepseek`、`-qwen`、`-mistral`、`-moonshot`、`-zhipu`、`-minimax`）。上述 feature 会引入对应的 crate，Cargo 只下载所选厂商的数据。只启用 OpenAI 时下载量约为 1.5 MB，全部启用时为 8.0 MB。

如果构建时未包含某份词表，`list_encodings()` 就不会列出对应的编码，`get_encoding()` 查询该编码时会返回 `None`。`pricing` 表独立于词表，只需估算费用时，可以关闭所有 `vocab-*` feature。

按厂商启用词表可使用 `vocab-openai`、`vocab-meta`、`vocab-deepseek`、`vocab-qwen`、`vocab-mistral`、`vocab-moonshot`、`vocab-zhipu` 或 `vocab-minimax`。默认启用的 `vocabs-all` 包含所有词表。下表列出了各份词表对应的 feature。

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

`数据`列表示启用该编码的 feature 后，词表会占用多少额外的二进制空间。多个编码共用同一份数据文件时，只计入一次。标有 `+ 基座` 的三份词表沿用基础词表的 token 排名，只需存储末尾新增的部分。

| 编码 | 厂商 | Feature | 数据 | 适用模型 |
|---|---|---|---|---|
| `o200k_base` | OpenAI | `vocab-o200k_base` | 815 KB | GPT-4o、GPT-4.1、GPT-4.5、GPT-5〜5.6（含 Sol/Terra/Luna）、GPT-6.x（推定，官方未公布）、o1、o3、o4-mini |
| `o200k_harmony` | OpenAI | `vocab-o200k_base` | — | gpt-oss（harmony 对话格式） |
| `cl100k_base` | OpenAI | `vocab-cl100k_base` | 373 KB | GPT-4、GPT-4 Turbo、GPT-3.5 Turbo、text-embedding-*、davinci-002、babbage-002 |
| `p50k_base` | OpenAI | `vocab-p50k_base` | 55 B + 基座 | text-davinci-002/003、code-davinci-*、code-cushman-* |
| `p50k_edit` | OpenAI | `vocab-p50k_base` | — | text-davinci-edit-*、code-davinci-edit-* |
| `r50k_base` | OpenAI | `vocab-r50k_base` | 182 KB | GPT-3 时代：davinci、curie、babbage、ada |
| `gpt2` | OpenAI | `vocab-r50k_base` | — | GPT-2（`r50k_base` 的别名） |
| `llama3` | Meta | `vocab-llama3` | 111 KB + 基座 | Llama 3、3.1、3.2、3.3 |
| `llama4` | Meta | `vocab-llama4` | 816 KB | Llama 4 Scout / Maverick、Muse |
| `deepseek_v3` | DeepSeek | `vocab-deepseek_v3` | 514 KB | DeepSeek V3、R1 |
| `deepseek_v4` | DeepSeek | `vocab-deepseek_v3` | — | DeepSeek V4 Pro / Flash、V4.1 Flash（V3 词表 + V4 特殊 token） |
| `qwen2` | 阿里巴巴 | `vocab-qwen2` | 564 KB | Qwen 2.5、Qwen 3 |
| `qwen3_5` | 阿里巴巴 | `vocab-qwen3_5` | 989 KB | Qwen 3.5〜3.8、Qwen 4（推定，官方未公布） |
| `mistral_v3` | Mistral | `vocab-mistral_v3` | 525 KB | Mistral、Mixtral、Large 3、Small 4（Tekken 分词器） |
| `mistral_v15` | Mistral | `vocab-mistral_v15` | 853 KB | Mistral Large 4（Tekken v15） |
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

// 多次调用共用一个缓冲区，每次调用不再分配内存
let mut ids = Vec::new();
enc.encode_into("hello world", &mut ids);         // 追加到 `ids` 末尾

// 单个 token 的字节，直接借用词表，不分配；累加长度即得每个 token 在原文中的字节位置
let hello: Option<&[u8]> = enc.token_bytes(ids[0]);
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

### 重复文本

`TextMemo` 缓存完整文本及其 token ID。它逐字节核对输入，命中后以切片形式返回已有的 ID；未命中时则编码并保存结果。首次编码时，3 KB 及以上的文本最多增加约 4% 的耗时，900 字节的文本最多增加一个计时器刻度（42 ns）。不创建 `TextMemo`，就不会保存任何文本和 id。

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let mut memo = enc.memo(64 << 20); // 文本和 id 合计最多约 64 MiB
let first = memo.encode("a long system prompt").to_vec();
assert_eq!(memo.encode("a long system prompt"), first); // 直接从 memo 取结果
```

### 并行编码

`encode`、`count`、`encode_with_special_tokens` 和 `count_with_special_tokens` 会把 32 KiB 及以上的文本分块，交给整个进程共用的 worker 线程池编码。首次处理这类输入时才创建线程池，输出与单线程编码完全一致。

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let text = "word ".repeat(20_000);
let tokens = enc.encode(&text); // 100 KB：多线程编码

// 每次调用可用的线程数，含调用方线程；1 = 单线程，0 = 恢复默认
tiktoken::set_parallelism(1);
assert_eq!(enc.encode(&text), tokens);
tiktoken::set_parallelism(0);
```

- 默认线程数等于性能核数量。如果系统不区分核心类型，则使用全部核心。可将环境变量 `TIKTOKEN_THREADS` 设为正整数来指定线程数
- 设置 `TIKTOKEN_THREADS=1` 或调用 `set_parallelism(1)` 可关闭并行编码。此后每次调用都在调用方线程上执行，不会启动 worker 线程
- 线程池一次只处理一个并行调用。如果线程池正忙，例如应用已经从多个线程发起编码，新调用会直接在调用方线程上执行，无需等待。这样进程中用于编码的线程数就不会超过设定值
- 为兼容已有代码，本库保留了 `encode_parallel`。它与 `encode` 的行为相同，无需启用 `parallel` feature
- `wasm32` 没有线程支持，所有调用都以单线程执行

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

费用估算支持 153 个模型，涵盖 OpenAI、Anthropic、Google、Meta、DeepSeek、阿里巴巴、Mistral、Moonshot、智谱、MiniMax 和 Voyage。

## WebAssembly

npm 提供 [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)。该包使用 WebAssembly SIMD，支持的最低浏览器版本为 Chrome 91、Firefox 89 和 Safari 16.4。

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

代码采用双重许可，可任选 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE)。

内嵌词表不在上述许可范围内，各自沿用发布方的许可证。[LICENSE-3RD-PARTY](LICENSE-3RD-PARTY) 列出了每份词表的来源和许可证，[`licenses/`](licenses/) 收录了许可证全文，[NOTICE](NOTICE) 列出了各许可证要求的署名。

部分词表有额外的使用条件。根据 Llama 许可证，月活跃用户超过 7 亿的产品须向 Meta 取得许可。Mistral 要求提供模型即服务（Model-as-a-Service）的企业取得 `mistral_v15` 的商业许可。Kimi 和 MiniMax 的许可证对超过一定规模的使用者有展示方面的要求。DeepSeek 的许可证对用途有限制。构建时关闭某份词表的 `vocab-*` feature，即可从产物中排除该词表。

Built with Llama.
