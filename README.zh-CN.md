# rust-tiktoken

[![tiktoken on crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust&label=tiktoken)](https://crates.io/crates/tiktoken)
[![tiktoken-wasm on npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm?style=flat-square&logo=npm&label=tiktoken-wasm)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-tiktoken/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-tiktoken/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#许可证)

[English](README.md) | **简体中文** | [日本語](README.ja.md) · [tiktoken.golia.jp](https://tiktoken.golia.jp)（在浏览器中试用）

> **从 3.5.x 升级**：3.6.0、3.8.0 和 5.0.0 修正了一些和参考分词器不一致的 token id，缓存的 token 数和存下来的 id 序列要重新算。详见 [CHANGELOG](tiktoken/CHANGELOG.md#upgrading-from-35x--token-id-changes)。
>
> **升级到 4.0**：token id 和 API 都没变。唯一不兼容的地方是词表：每份词表现在由一个默认开启的 `vocab-*` feature 控制，依赖里写了 `default-features = false` 的，要把用到的词表列出来。
>
> **升级到 5.2**：API 没变。32 KiB 及以上的文本默认用多个线程编码，`tiktoken::set_parallelism(1)` 或 `TIKTOKEN_THREADS=1` 可以关掉。wasm 包需要浏览器支持 WebAssembly SIMD（Chrome 91、Firefox 89、Safari 16.4）。有 336 个码点只在较新的 Unicode 版本里算数字，它们在 `mistral_v15`、`deepseek_v3` / `deepseek_v4` 上的切分方式变了。详见 [CHANGELOG](tiktoken/CHANGELOG.md#520---2026-11-23)。
>
> **升级到 5.0**：有两处不兼容。一是定价相关的结构体和 `VisionPricing` 加了 `#[non_exhaustive]`，`match` 要加 `_` 分支，也不能再用字面量构造这些结构体。二是 Llama 4、Qwen 3.5〜3.8、Mistral Large 4 和 `deepseek-flash` 改为对应这些模型实际用的词表，它们的 token 数会变。5.1 把词表数据挪到了按厂商划分的 `tiktoken-vocab-*` crate，其他没变。

rust-tiktoken 提供速度最快的 Rust BPE 分词器及其 WebAssembly 绑定，可直接替代 OpenAI [tiktoken](https://github.com/openai/tiktoken) 和主流开放模型的分词器（Llama 3、DeepSeek、Qwen、Mistral、Kimi、GLM、MiniMax）。分词器用手写扫描器处理 ASCII 和 CJK，按键长组织词表，并缓存整个片段的编码结果。原生编码速度是 tiktoken-rs 的 22〜83 倍、fastokens 的 1.5〜12 倍。浏览器中是 gpt-tokenizer 的 5.8〜7.0 倍。中文、日文文章上同样领先。

## workspace 中的 crate

| 路径 | crate 或包 | 说明 | 版本 |
|:-----|:----------------|:-----|:-----|
| [`tiktoken/`](tiktoken/) | [`tiktoken`](https://crates.io/crates/tiktoken) | Rust BPE 分词器，支持 20 种编码、153 个模型及多家厂商的定价 | [![crates.io](https://img.shields.io/crates/v/tiktoken.svg?style=flat-square)](https://crates.io/crates/tiktoken) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`tiktoken-wasm`](https://crates.io/crates/tiktoken-wasm)（Rust） | 分词器的 wasm 绑定 crate | [![crates.io](https://img.shields.io/crates/v/tiktoken-wasm.svg?style=flat-square)](https://crates.io/crates/tiktoken-wasm) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)（npm） | 同一套 wasm 绑定，通过 `wasm-pack` 发布到 npm | [![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm.svg?style=flat-square)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) |

> 两个 crate 共用一个 workspace，每次发布都同步更新版本号，并以同一版本号发布。

## 主要特性

- **运行时不用正则**：每种编码的切分规则都由查表的扫描器实现，表由 Unicode 数据生成。cl100k 和 o200k 系列用 SIMD（NEON、SSE2 或 AVX2、wasm simd128）一次判断 64 字节里哪些位置是片段起点。`regex` crate 只在测试里用来核对扫描器的结果
- **20 种编码、153 个模型、11 家厂商**：OpenAI（GPT-4/4o/4.1/4.5、GPT-5.x、GPT-6.x、o1/o3/o4-mini、gpt-oss）、Llama 3/4、DeepSeek V3/V4、Qwen 2〜3.8、Mistral（含 Large 4）、Kimi K2/K3、GLM-4/5、MiniMax M2/M3。另外可以按美元估算费用，估算还覆盖 Anthropic 和 Google 的模型
- **重复文本**：`enc.memo(max_bytes)` 返回一个 `TextMemo`，需要时才创建。遇到见过的文本，它直接从内存给出结果：一本 200 KB 的书重新编码要 110 µs，从 `TextMemo` 取只要 3.5 µs。`encode` 本身什么都不保存
- **小而可移植**：20 份词表一共 7.8 MB，编译时内嵌，每份都可以单独去掉。只要 cl100k 的构建只带 373 KB 词表数据，cargo 也只下载你启用的厂商的数据。词表按 key 的长度分层，整个片段的编码结果会被缓存，BPE 合并在线性扫描和堆之间切换。长文本由内置线程池分给多个线程编码，线程池只用标准库。`count()` 不分配内存。全部用 Rust 写成，没有 C 依赖，运行时只依赖 `ruzstd` 和词表数据 crate。wasm 构建不依赖其他文件

各 crate 的 README 提供完整的 API 文档、支持的模型列表和基准测试结果，见 [`tiktoken/`](tiktoken/README.md) 和 [`tiktoken-wasm/`](tiktoken-wasm/README.md)。

## 快速开始

### Rust

```toml
[dependencies]
tiktoken = "5"
```

```rust
// 按编码名获取
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode("hello world");
assert_eq!(enc.decode_to_string(&tokens).unwrap(), "hello world");

// 不分配 token 向量的计数
let n = enc.count("The quick brown fox.");

// 多次调用共用一个缓冲区，encode_into 追加到末尾
let mut ids = Vec::new();
enc.encode_into("hello world", &mut ids);

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

在 Apple M4 Pro Mac mini 上单线程运行 `encode`，速度是 tiktoken-rs 的 22〜83 倍（ASCII 文本 22〜83 倍，中日文文章 50〜82 倍，没有重复片段的 CJK 对抗语料 36 倍），是 Python tiktoken 的 43〜92 倍。和 fastokens 比，单线程快 1.5〜12 倍；用默认线程数编码一本 200 KB 的书，快 3 倍。在浏览器里（wasm），速度是 gpt-tokenizer 的 5.8〜7.0 倍、tiktoken-fast 的 1.1〜2.5 倍。完整数据和测试方法见 [`tiktoken/README.md#performance`](tiktoken/README.md#performance)。

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

`tiktoken` 和 `tiktoken-wasm` 使用相同的版本号，用 git-flow 一起发布，不走 PR：

```bash
# 发布闸门：完整属性扫描（约 35 分钟；CI 只跑抽样）
PROPTEST_CASES=25000 cargo test --workspace --all-targets

git flow release start X.Y.Z
# 把版本升到 X.Y.Z：tiktoken/Cargo.toml、tiktoken-wasm/Cargo.toml
#（及其 tiktoken path 依赖）；确定两个 CHANGELOG。
git flow release finish X.Y.Z                       # 合并到 master、打 vX.Y.Z tag、回合 develop
git tag -a tiktoken-wasm-vX.Y.Z vX.Y.Z^{commit} -m "tiktoken-wasm X.Y.Z"
git push origin master develop vX.Y.Z tiktoken-wasm-vX.Y.Z
# tag `v*` 发布 tiktoken crate；`tiktoken-wasm-v*` 发布 wasm crate + npm
```

测试以 `regex` crate 为参照，核对扫描器的输出。测试包括 7 项预分词属性测试（每项都覆盖所有切分规则系列）、短字符串穷举测试和 12 项编解码往返属性测试。`PROPTEST_CASES` 控制每次运行时生成的随机用例数。无论取值多少，测试都会先重放 `proptest-regressions/` 中保存的种子，再运行随机用例，确保每次都检查曾经出错的输入。

## 许可证

代码采用双重许可，可任选 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE)。

内嵌词表不在上述许可范围内，各自沿用发布方的许可证。[LICENSE-3RD-PARTY](tiktoken/LICENSE-3RD-PARTY) 列出了每份词表的来源和许可证，[`licenses/`](tiktoken/licenses/) 收录了许可证全文，[NOTICE](tiktoken/NOTICE) 列出了各许可证要求的署名。

部分词表有额外的使用条件。根据 Llama 许可证，月活跃用户超过 7 亿的产品须向 Meta 取得许可。Mistral 要求提供模型即服务（Model-as-a-Service）的企业取得 `mistral_v15` 的商业许可。Kimi 和 MiniMax 的许可证对超过一定规模的使用者有展示方面的要求。DeepSeek 的许可证对用途有限制。构建时关闭某份词表的 `vocab-*` feature，即可从产物中排除该词表。

Built with Llama.
