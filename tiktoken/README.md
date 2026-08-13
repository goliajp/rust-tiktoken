# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#license)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square&logo=rust)](Cargo.toml)
[![Downloads](https://img.shields.io/crates/d/tiktoken?style=flat-square)](https://crates.io/crates/tiktoken)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

The fastest Rust BPE tokenizer — 5–49x faster than tiktoken-rs natively (15–17x on Chinese and Japanese prose) and 2–4x faster than gpt-tokenizer in the browser via wasm. Hand-written scanners cover ASCII and CJK alike, the vocabulary is layered by key size, and repeated pieces are memoised whole. Compatible with OpenAI [tiktoken](https://github.com/openai/tiktoken) and supports **all mainstream LLM tokenizers** — OpenAI, Llama 3, DeepSeek, Qwen, Mistral, Kimi, GLM, MiniMax.

## Features

- **Multi-provider**: 17 encodings across 8 vendors (OpenAI, Meta, DeepSeek, Alibaba, Mistral, Moonshot, Zhipu, MiniMax)
- **Fast**: hand-written pre-tokenizer for ASCII and CJK (bypasses the regex), key-size-layered vocabulary, whole-piece memoisation, hybrid BPE merge
- **Parallel encoding**: optional rayon-based multi-threaded encoding for large texts
- **Pricing**: cost estimation for 107 models across 10 providers
- **Compact**: all 17 vocabularies embedded in 5.1 MB, and opt-out per vocabulary — a cl100k-only build carries 373 KB of data
- **Zero-alloc counting**: `count()` path avoids token vector allocation

## Performance

Token outputs are asserted identical across implementations before anything is
timed; each figure is one full pass over the corpus, median of 9 rounds after
warmup. Corpora are byte-identical in every harness (`bench-compare/`,
`benches/bench_python.py`, `web/bench/`).

#### Native — Apple M4 Mac mini, single thread, `encode`

`cargo run --release -p bench-compare`

| Corpus | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken** | vs rs | vs Python |
|---|---|---|---|---|---|
| short (13 B) | 1.6 µs | 1,081 ns | **33 ns** | **33x** | **48x** |
| medium (900 B) | 31.9 µs | 52.2 µs | **1.1 µs** | **47x** | **29x** |
| English prose (45 KB) | 1,500 µs | 2,498 µs | **51.5 µs** | **49x** | **29x** |
| Chinese prose (4.3 KB) | 119.8 µs | 134.7 µs | **8.1 µs** | **17x** | **15x** |
| Japanese prose (4.6 KB) | 131.0 µs | 144.6 µs | **8.6 µs** | **17x** | **15x** |
| mixed CJK ×50 (4.5 KB) | 138.9 µs | 160.3 µs | **15.2 µs** | **11x** | **9.2x** |
| adversarial CJK, no repeats (3.9 KB) | 131.7 µs | 141.2 µs | **25.9 µs** | **5.5x** | **5.1x** |
| code (3.9 KB) | 263.7 µs | 317.7 µs | **11.1 µs** | **29x** | **24x** |

o200k_base tracks the same ratios (5–48x vs tiktoken-rs). `count()` runs
another 5–15% faster than `encode` — it never allocates the id vector.

#### In the browser — Mac Studio (M4 Max), Chromium

`npm run bench` in `web/` — this crate compiled to wasm, against the two
mainstream JavaScript tokenizers.

| Corpus | gpt-tokenizer 3.4 | js-tiktoken 1.0 | **tiktoken (wasm)** |
|---|---|---|---|
| Chinese prose (4.3 KB) | 36.8 µs | 8,029 µs | **13.4 µs** |
| Japanese prose (4.6 KB) | 27.4 µs | 15,862 µs | **13.5 µs** |
| mixed CJK ×50 (4.5 KB) | 41.2 µs | 4,665 µs | **24.2 µs** |
| adversarial CJK, no repeats (3.9 KB) | 49.6 µs | 3,832 µs | **40.3 µs** |
| English prose (45 KB) | 478 µs | 7,010 µs | **112.5 µs** |
| code (3.9 KB) | 76.0 µs | 916 µs | **19.5 µs** |

The adversarial corpus never repeats a piece, which disables every
implementation's memoisation — it is the floor, and the lead holds there too.

<details>
<summary>Why is it faster?</summary>

| | tiktoken | tiktoken-rs | Python tiktoken |
|---|---|---|---|
| Pre-tokenize | Hand-written scanners for ASCII **and** CJK (Han/kana/hangul, fullwidth forms); regex only as arbiter and rare-case fallback | always runs the regex | always runs the regex |
| Regex engine (fallback) | `regex` (DFA, linear time) | `fancy-regex` (backtracking) | `regex` via PyO3 + FFI overhead |
| Vocab lookup | Layered by key size: direct-indexed tables for 1–2-byte keys, open addressing with inlined keys for 3–8, tagged arena slots above | `HashMap<Vec<u8>>` (200k allocs) | Rust `HashMap` behind PyO3 |
| Repeated pieces | Thread-local direct-mapped memoisation, byte keys | none | none |
| BPE merge | Hybrid: stack linear-scan (short pieces) + heap (long) | O(n*m) linear scan | O(n*m) linear scan |
| `count()` without alloc | yes | no | no |

The scanners are pinned to the regex by property tests (hundreds of thousands
of random inputs per run), and every encoding is pinned to its vendor's
tokenizer by 44,518 differential fixture cases.

Benchmark source: [`benches/`](benches/), [`../bench-compare/`](../bench-compare/). Reproducible via `cargo bench` / `cargo run -p bench-compare`.

</details>

## Installation

```toml
[dependencies]
tiktoken = "4"

# optional: multi-threaded encoding for large texts
tiktoken = { version = "4", features = ["parallel"] }
```

### Picking vocabularies

All 17 encodings are on by default (5.1 MB of vocabulary data). Turn off the
defaults and name what you use to carry only that:

```toml
# GPT-4o / GPT-5 only — 815 KB of vocabulary data instead of 5.1 MB
tiktoken = { version = "4", default-features = false, features = ["vocab-o200k_base"] }

# a whole vendor
tiktoken = { version = "4", default-features = false, features = ["vocab-openai"] }
```

Measured on `examples/count_tokens` (release): 6,480,912 bytes with every
vocabulary, 2,226,704 with `vocab-cl100k_base` alone.

An encoding whose vocabulary is not compiled in is simply absent —
`list_encodings()` does not list it and `get_encoding()` returns `None`. The
`pricing` tables are independent of vocabularies, so a build with no `vocab-*`
feature at all is valid if you only need cost estimation.

Vendor groups: `vocab-openai`, `vocab-meta`, `vocab-deepseek`, `vocab-qwen`,
`vocab-mistral`, `vocab-moonshot`, `vocab-zhipu`, `vocab-minimax`, and
`vocabs-all` (the default). Per-vocabulary features are listed in the table
below.

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

`Data` is the vocabulary bytes the encoding's feature adds to your binary.
Vocabularies that share a data file cost nothing the second time, and the three
marked `+ base` are rank-aligned extensions storing only their tail.

| Encoding | Provider | Feature | Data | Models |
|---|---|---|---|---|
| `o200k_base` | OpenAI | `vocab-o200k_base` | 815 KB | GPT-4o, GPT-4.1, GPT-4.5, GPT-5–5.6 (incl. Sol/Terra/Luna), o1, o3, o4-mini |
| `o200k_harmony` | OpenAI | `vocab-o200k_base` | — | gpt-oss (harmony chat format) |
| `cl100k_base` | OpenAI | `vocab-cl100k_base` | 373 KB | GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, text-embedding-*, davinci-002, babbage-002 |
| `p50k_base` | OpenAI | `vocab-p50k_base` | 55 B + base | text-davinci-002/003, code-davinci-*, code-cushman-* |
| `p50k_edit` | OpenAI | `vocab-p50k_base` | — | text-davinci-edit-*, code-davinci-edit-* |
| `r50k_base` | OpenAI | `vocab-r50k_base` | 182 KB | GPT-3 era: davinci, curie, babbage, ada |
| `gpt2` | OpenAI | `vocab-r50k_base` | — | GPT-2 (alias for `r50k_base`) |
| `llama3` | Meta | `vocab-llama3` | 111 KB + base | Llama 3, 3.1, 3.2, 3.3, 4 |
| `deepseek_v3` | DeepSeek | `vocab-deepseek_v3` | 514 KB | DeepSeek V3, R1 |
| `deepseek_v4` | DeepSeek | `vocab-deepseek_v3` | — | DeepSeek V4 Pro / Flash (V3 vocab + V4 special tokens) |
| `qwen2` | Alibaba | `vocab-qwen2` | 564 KB | Qwen 2.5, Qwen 3 |
| `mistral_v3` | Mistral | `vocab-mistral_v3` | 525 KB | Mistral, Mixtral (Tekken tokenizer) |
| `kimi_k2` | Moonshot | `vocab-kimi_k2` | 659 KB | Kimi K2 / K2.5 / K2.6 |
| `kimi_k3` | Moonshot | `vocab-kimi_k2` | — | Kimi K3 (K2 vocab + K3 special tokens) |
| `glm4` | Zhipu | `vocab-glm4` | 578 KB | GLM-4.5 / 4.6 / 4.7 |
| `glm5` | Zhipu | `vocab-glm5` | 6 KB + base | GLM-5 / 5.2 |
| `minimax_m2` | MiniMax | `vocab-minimax_m2` | 822 KB | MiniMax M2 / M2.1 / M2.5 / M2.7 |

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

Supports 107 models across OpenAI, Anthropic, Google, Meta, DeepSeek, Alibaba, and Mistral.

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
