# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#license)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square&logo=rust)](Cargo.toml)
[![Downloads](https://img.shields.io/crates/d/tiktoken?style=flat-square)](https://crates.io/crates/tiktoken)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

The fastest Rust BPE tokenizer — 15–40x faster than tiktoken-rs on ASCII text (≈2x on CJK/Unicode), thanks to a hand-written ASCII fast-path. Compatible with OpenAI [tiktoken](https://github.com/openai/tiktoken) and supports **all mainstream LLM tokenizers** — OpenAI, Llama 3, DeepSeek, Qwen, and Mistral.

## Features

- **Multi-provider**: 11 encodings across 5 vendors (OpenAI, Meta, DeepSeek, Alibaba, Mistral)
- **Fast**: hand-written ASCII fast-path pre-tokenizer (bypasses the regex), arena-based vocabulary, hybrid BPE merge
- **Parallel encoding**: optional rayon-based multi-threaded encoding for large texts
- **Pricing**: cost estimation for 68 models across 7 providers
- **Compact**: ruzstd-compressed vocabulary data embedded at compile time
- **Zero-alloc counting**: `count()` path avoids token vector allocation

## Performance

All benchmarks on Apple M4 Mac mini, single-threaded. Token output verified identical across all three implementations.

#### cl100k_base encode

| Input | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken** | vs tiktoken-rs | vs Python |
|---|---|---|---|---|---|
| short (13 B) | 1,700 ns | 1,248 ns | **43 ns** | **29x** | **40x** |
| medium (900 B) | 32.2 us | 53.8 us | **1.5 us** | **35x** | **21x** |
| long (45 KB) | 1,500 us | 2,611 us | **74 us** | **35x** | **20x** |
| unicode (4.5 KB) | 141 us | 164 us | **91 us** | **1.8x** | **1.6x** |
| code (3.9 KB) | 247 us | 264 us | **17 us** | **16x** | **15x** |

#### o200k_base encode

| Input | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken** | vs tiktoken-rs | vs Python |
|---|---|---|---|---|---|
| short (13 B) | 1,600 ns | 1,051 ns | **40 ns** | **26x** | **40x** |
| medium (900 B) | 58.3 us | 56.2 us | **1.5 us** | **37x** | **39x** |
| long (45 KB) | 2,900 us | 2,799 us | **73 us** | **38x** | **40x** |
| unicode (4.5 KB) | 204 us | 187 us | **95 us** | **2.0x** | **2.2x** |
| code (3.9 KB) | 332 us | 253 us | **16 us** | **16x** | **21x** |

<details>
<summary>Why is it faster?</summary>

| | tiktoken | tiktoken-rs | Python tiktoken |
|---|---|---|---|
| Vocab storage | Arena-based (single alloc, cache-friendly) | `HashMap<Vec<u8>>` (200k allocs) | Rust `HashMap` behind PyO3 |
| Pre-tokenize (ASCII) | Hand-written ASCII fast-path, skips the regex | always runs the regex | always runs the regex |
| Regex engine (fallback) | `regex` (DFA, linear time) | `fancy-regex` (backtracking) | `regex` via PyO3 + FFI overhead |
| Hash map | Custom open-addressing + `FxHash` | `rustc-hash` v1 | standard `HashMap` |
| BPE merge | Hybrid: stack linear-scan (short pieces) + heap (long) | O(n*m) linear scan | O(n*m) linear scan |
| `count()` without alloc | yes | no | no |

Benchmark source: [`benches/`](benches/). Reproducible via `cargo bench`.

</details>

## Installation

```toml
[dependencies]
tiktoken = "3"

# optional: multi-threaded encoding for large texts
tiktoken = { version = "3", features = ["parallel"] }
```

## Quick Start

```rust
// by encoding name
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode("hello world");
let text = enc.decode_to_string(&tokens).unwrap();
assert_eq!(text, "hello world");

// by model name — works across all providers
let enc = tiktoken::encoding_for_model("gpt-4o").unwrap();
let count = enc.count("hello world"); // zero-alloc fast path

let enc = tiktoken::encoding_for_model("llama-3.1-70b").unwrap();
let enc = tiktoken::encoding_for_model("deepseek-v3").unwrap();
let enc = tiktoken::encoding_for_model("qwen2.5-72b").unwrap();
```

## Supported Encodings

| Encoding | Provider | Models |
|---|---|---|
| `o200k_base` | OpenAI | GPT-4o, GPT-4.1, GPT-4.5, GPT-5, o1, o3, o4-mini |
| `o200k_harmony` | OpenAI | gpt-oss (harmony chat format) |
| `cl100k_base` | OpenAI | GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, text-embedding-*, davinci-002, babbage-002 |
| `p50k_base` | OpenAI | text-davinci-002/003, code-davinci-*, code-cushman-* |
| `p50k_edit` | OpenAI | text-davinci-edit-*, code-davinci-edit-* |
| `r50k_base` | OpenAI | GPT-3 era: davinci, curie, babbage, ada |
| `gpt2` | OpenAI | GPT-2 (alias for `r50k_base`) |
| `llama3` | Meta | Llama 3, 3.1, 3.2, 3.3, 4 |
| `deepseek_v3` | DeepSeek | DeepSeek V3, R1 |
| `qwen2` | Alibaba | Qwen 2.5, Qwen 3 |
| `mistral_v3` | Mistral | Mistral, Mixtral (Tekken tokenizer) |

## API

### Encode / decode

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();

let tokens = enc.encode("hello world");           // Vec<u32>
let text = enc.decode_to_string(&tokens).unwrap(); // String
let bytes = enc.decode(&tokens);                   // Vec<u8>
```

### Special tokens

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode_with_special_tokens("hello<|endoftext|>world");
// tokens will include the special token id for <|endoftext|>
```

### Count tokens

```rust
let enc = tiktoken::get_encoding("o200k_base").unwrap();
let count = enc.count("The quick brown fox jumps over the lazy dog.");
// faster than encode().len() -- no token vector allocated

// count with special token recognition
let count = enc.count_with_special_tokens("hello<|endoftext|>world");
```

### Parallel encoding

```rust
// requires `parallel` feature
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode_parallel("...very long text...");
// identical output, uses rayon for texts >= 4KB
```

### Cost estimation

```rust
use tiktoken::pricing;

// quick estimate
let cost = pricing::estimate_cost("gpt-4o", 1_000_000, 500_000).unwrap();

// with prompt caching
let model = pricing::get_model("claude-opus-4").unwrap();
let cost = model.estimate_cost_with_cache(500_000, 500_000, 200_000);

// list all models for a provider
let models = pricing::models_by_provider(pricing::Provider::DeepSeek);
```

Supports 68 models across OpenAI, Anthropic, Google, Meta, DeepSeek, Alibaba, and Mistral.

## WebAssembly

Available as [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) on npm.

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

enc.free() // release WASM memory when done
```

| Bundler | Plugin |
|---------|--------|
| Vite | [vite-plugin-wasm](https://www.npmjs.com/package/vite-plugin-wasm) + [vite-plugin-top-level-await](https://www.npmjs.com/package/vite-plugin-top-level-await) |
| webpack 5 | Built-in `asyncWebAssembly` experiment |
| Next.js | [next.config.js `webpack.experiments`](https://nextjs.org/docs/app/api-reference/next-config-js/webpack) |

See [`examples/react-app`](examples/react-app/) for a complete demo.

<details>
<summary>Build from source</summary>

```bash
# requires wasm-pack: cargo install wasm-pack
cd tiktoken-wasm
wasm-pack build --target web --release
```

</details>

<!-- ECOSYSTEM BEGIN (generated — edit ecosystem.toml, not this block) -->

## Ecosystem

**tiktoken** · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · [chunkedrs](https://crates.io/crates/chunkedrs) · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

Third-party vocabulary data licenses: see [LICENSE-3RD-PARTY](LICENSE-3RD-PARTY).
