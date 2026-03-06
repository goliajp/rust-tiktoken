# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen?style=flat-square)](src/)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-blue?style=flat-square&logo=rust)](Cargo.toml)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

The fastest Rust BPE tokenizer compatible with OpenAI's [tiktoken](https://github.com/openai/tiktoken). **7–10x faster than tiktoken-rs**, **4–15x faster than Python tiktoken**.

## Performance

All benchmarks on Apple M4 Mac mini, single-threaded. Token output verified identical across all three implementations.

#### cl100k_base encode

| Input | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken 2.1** | vs tiktoken-rs | vs Python |
|---|---|---|---|---|---|
| short (13 B) | 1,700 ns | 1,248 ns | **118 ns** | **10.5x** | **14x** |
| medium (900 B) | 32.2 µs | 53.8 µs | **7.3 µs** | **7.3x** | **4.4x** |
| long (45 KB) | 1,500 µs | 2,611 µs | **373 µs** | **7.0x** | **4.0x** |
| unicode (4.5 KB) | 141 µs | 164 µs | **97 µs** | **1.7x** | **1.5x** |
| code (3.9 KB) | 247 µs | 264 µs | **34 µs** | **7.7x** | **7.3x** |

#### o200k_base encode

| Input | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken 2.1** | vs tiktoken-rs | vs Python |
|---|---|---|---|---|---|
| short (13 B) | 1,600 ns | 1,051 ns | **116 ns** | **9.1x** | **14x** |
| medium (900 B) | 58.3 µs | 56.2 µs | **7.3 µs** | **7.7x** | **8.0x** |
| long (45 KB) | 2,900 µs | 2,799 µs | **374 µs** | **7.5x** | **7.8x** |
| unicode (4.5 KB) | 204 µs | 187 µs | **104 µs** | **1.8x** | **2.0x** |
| code (3.9 KB) | 332 µs | 253 µs | **33 µs** | **7.6x** | **10x** |

<details>
<summary>Why is it faster?</summary>

| | tiktoken | tiktoken-rs | Python tiktoken |
|---|---|---|---|
| Regex engine | `regex` (DFA, linear time) | `fancy-regex` (backtracking) | `regex` via PyO3 + FFI overhead |
| Hash map | `FxHashMap` (fast for small keys) | `rustc-hash` v1 | Rust `HashMap` behind PyO3 |
| BPE merge | rank-cached (2 neighbor recompute) | standard | standard |
| `count()` without alloc | yes | no | no |
| Runtime dependencies | 4 crates | 16 crates | — |

Benchmark source: [`benches/`](benches/). Reproducible via `cargo bench`.

</details>

## Installation

```toml
[dependencies]
tiktoken = "2.1"
```

## Quick Start

```rust
// by encoding name
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode("hello world");
let text = enc.decode_to_string(&tokens).unwrap();
assert_eq!(text, "hello world");

// by model name
let enc = tiktoken::encoding_for_model("gpt-4o").unwrap();
let count = enc.count("hello world"); // zero-alloc fast path
```

## Supported Encodings

| Encoding | Models |
|---|---|
| `o200k_base` | GPT-4o, GPT-4o-mini, o1, o3, o4-mini |
| `cl100k_base` | GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, text-embedding-ada-002, text-embedding-3-* |
| `p50k_base` | text-davinci-002/003, code-davinci-*, code-cushman-* |
| `p50k_edit` | text-davinci-edit-*, code-davinci-edit-* |
| `r50k_base` | text-davinci-001, text-curie, text-babbage, text-ada, davinci, curie, babbage, ada |

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
// faster than encode().len() — no token vector allocated
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
let models = pricing::models_by_provider(pricing::Provider::Google);
```

Supports 26 models across OpenAI, Anthropic Claude, and Google Gemini.

## WebAssembly

Available as [`@goliapkg/tokenrs-wasm`](https://www.npmjs.com/package/@goliapkg/tokenrs-wasm) on npm.

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
