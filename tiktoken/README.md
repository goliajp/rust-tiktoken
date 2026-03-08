# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square&logo=rust)](Cargo.toml)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

The fastest Rust BPE tokenizer. Compatible with OpenAI [tiktoken](https://github.com/openai/tiktoken) and supports **all mainstream LLM tokenizers** — OpenAI, Llama 3, DeepSeek, Qwen, and Mistral.

## Features

- **Multi-provider**: 9 encodings across 5 vendors (OpenAI, Meta, DeepSeek, Alibaba, Mistral)
- **Fast**: arena-based vocabulary, heap-accelerated BPE merge, DFA regex
- **Parallel encoding**: optional rayon-based multi-threaded encoding for large texts
- **Pricing**: cost estimation for 57 models across 7 providers
- **Compact**: ruzstd-compressed vocabulary data embedded at compile time
- **Zero-alloc counting**: `count()` path avoids token vector allocation

## Performance

All benchmarks on Apple M4 Mac mini, single-threaded. Token output verified identical across all three implementations.

#### cl100k_base encode

| Input | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken 3.1** | vs tiktoken-rs | vs Python |
|---|---|---|---|---|---|
| short (13 B) | 1,700 ns | 1,248 ns | **118 ns** | **10.6x** | **14x** |
| medium (900 B) | 32.2 us | 53.8 us | **7.2 us** | **7.5x** | **4.5x** |
| long (45 KB) | 1,500 us | 2,611 us | **366 us** | **7.1x** | **4.1x** |
| unicode (4.5 KB) | 141 us | 164 us | **101 us** | **1.6x** | **1.4x** |
| code (3.9 KB) | 247 us | 264 us | **42 us** | **6.3x** | **5.9x** |

#### o200k_base encode

| Input | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken 3.1** | vs tiktoken-rs | vs Python |
|---|---|---|---|---|---|
| short (13 B) | 1,600 ns | 1,051 ns | **115 ns** | **9.1x** | **14x** |
| medium (900 B) | 58.3 us | 56.2 us | **7.1 us** | **7.9x** | **8.2x** |
| long (45 KB) | 2,900 us | 2,799 us | **365 us** | **7.7x** | **7.9x** |
| unicode (4.5 KB) | 204 us | 187 us | **99 us** | **1.9x** | **2.1x** |
| code (3.9 KB) | 332 us | 253 us | **41 us** | **6.2x** | **8.1x** |

<details>
<summary>Why is it faster?</summary>

| | tiktoken | tiktoken-rs | Python tiktoken |
|---|---|---|---|
| Vocab storage | Arena-based (single alloc, cache-friendly) | `HashMap<Vec<u8>>` (200k allocs) | Rust `HashMap` behind PyO3 |
| Regex engine | `regex` (DFA, linear time) | `fancy-regex` (backtracking) | `regex` via PyO3 + FFI overhead |
| Hash map | Custom open-addressing + `FxHash` | `rustc-hash` v1 | standard `HashMap` |
| BPE merge | Heap-accelerated O(n log n) | O(n*m) linear scan | O(n*m) linear scan |
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
| `o200k_base` | OpenAI | GPT-4o, GPT-4o-mini, o1, o3, o4-mini |
| `cl100k_base` | OpenAI | GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, text-embedding-* |
| `p50k_base` | OpenAI | text-davinci-002/003, code-davinci-*, code-cushman-* |
| `p50k_edit` | OpenAI | text-davinci-edit-*, code-davinci-edit-* |
| `r50k_base` | OpenAI | GPT-3 era: davinci, curie, babbage, ada |
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

Supports 57 models across OpenAI, Anthropic, Google, Meta, DeepSeek, Alibaba, and Mistral.

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

## License

[MIT](LICENSE)

Third-party vocabulary data licenses: see [LICENSE-3RD-PARTY](LICENSE-3RD-PARTY).
