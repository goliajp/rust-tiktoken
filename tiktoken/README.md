# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#license)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square&logo=rust)](Cargo.toml)
[![Downloads](https://img.shields.io/crates/d/tiktoken?style=flat-square)](https://crates.io/crates/tiktoken)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

The fastest Rust BPE tokenizer — 3.7–20x faster than tiktoken-rs natively (14–18x on Chinese and Japanese prose) and 1.1–2.7x faster than gpt-tokenizer in the browser via wasm. Hand-written scanners cover ASCII and CJK alike, the vocabulary is layered by key size, and repeated pieces are memoised whole. Compatible with OpenAI [tiktoken](https://github.com/openai/tiktoken) and supports **all mainstream LLM tokenizers** — OpenAI, Llama 3, DeepSeek, Qwen, Mistral, Kimi, GLM, MiniMax.

## Features

- **Multi-provider**: 20 encodings across 8 vendors (OpenAI, Meta, DeepSeek, Alibaba, Mistral, Moonshot, Zhipu, MiniMax)
- **Fast**: hand-written pre-tokenizer for ASCII and CJK (bypasses the regex), key-size-layered vocabulary, whole-piece memoisation, hybrid BPE merge
- **Parallel encoding**: optional rayon-based multi-threaded encoding for large texts
- **Pricing**: cost estimation for 153 models across 11 providers
- **Compact**: all 20 vocabularies embedded in 7.8 MB, and opt-out per vocabulary — a cl100k-only build carries 373 KB of data, and cargo downloads only the vendors you enable
- **Zero-alloc counting**: `count()` path avoids token vector allocation

## Performance

Token outputs are asserted identical across implementations before anything is
timed; each figure is one full pass over the corpus, median of 9 rounds after
warmup. Corpora are byte-identical in every harness (`bench-compare/`,
`benches/bench_python.py`, `web/bench/`).

#### Native — Apple M4 Mac mini, single thread, `encode`

`cargo run --release -p bench-compare`

| Corpus | Python tiktoken 0.14 | tiktoken-rs 0.12 | **tiktoken** | vs rs | vs Python |
|---|---|---|---|---|---|
| short (13 B) | 1.3 µs | 581 ns | **29 ns** | **20x** | **45x** |
| medium (900 B) | 23.3 µs | 19.7 µs | **1.1 µs** | **18x** | **21x** |
| English prose (45 KB) | 1,100 µs | 956 µs | **52.6 µs** | **18x** | **21x** |
| Chinese prose (4.3 KB) | 114.0 µs | 96.1 µs | **6.7 µs** | **14x** | **17x** |
| Japanese prose (4.6 KB) | 131.3 µs | 113.9 µs | **6.5 µs** | **18x** | **20x** |
| mixed CJK ×50 (4.5 KB) | 122.1 µs | 106.7 µs | **15.4 µs** | **6.9x** | **7.9x** |
| adversarial CJK, no repeats (3.9 KB) | 113.8 µs | 92.5 µs | **24.7 µs** | **3.7x** | **4.6x** |
| code (3.9 KB) | 180.0 µs | 159.4 µs | **11.0 µs** | **15x** | **16x** |

o200k_base runs 2.8–14x faster than tiktoken-rs. `count()` runs another
6–35% faster than `encode` — it never allocates the id vector.

#### In the browser — Apple M4 Mac mini, Chromium

`npm run bench` in `web/` — this crate compiled to wasm, against the two
mainstream JavaScript tokenizers.

| Corpus | gpt-tokenizer 4.1 | js-tiktoken 1.0 | **tiktoken (wasm)** |
|---|---|---|---|
| Chinese prose (4.3 KB) | 33.0 µs | 6,669 µs | **12.9 µs** |
| Japanese prose (4.6 KB) | 29.4 µs | 12,885 µs | **13.9 µs** |
| mixed CJK ×50 (4.5 KB) | 40.8 µs | 3,917 µs | **23.4 µs** |
| adversarial CJK, no repeats (3.9 KB) | 43.6 µs | 2,741 µs | **38.4 µs** |
| English prose (45 KB) | 276 µs | 7,522 µs | **102.0 µs** |
| code (3.9 KB) | 32.0 µs | 1,020 µs | **17.5 µs** |

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
tokenizer by 53,275 differential fixture cases.

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

All 20 encodings are on by default (7.8 MB of vocabulary data). Turn off the
defaults and name what you use to carry only that:

```toml
# GPT-4o / GPT-5 only — 815 KB of vocabulary data instead of 7.8 MB
tiktoken = { version = "5", default-features = false, features = ["vocab-o200k_base"] }

# a whole vendor
tiktoken = { version = "5", default-features = false, features = ["vocab-openai"] }
```

Measured on `examples/count_tokens` (release, Linux x86_64): 9,463,280 bytes
with every vocabulary, 2,550,256 with `vocab-cl100k_base` alone.

The data lives in one crate per vendor (`tiktoken-vocab-openai`, `-meta`,
`-deepseek`, `-qwen`, `-mistral`, `-moonshot`, `-zhipu`, `-minimax`), pulled in
by the features above, so cargo only downloads the vendors you enable: about
1.5 MB for OpenAI alone instead of 7.9 MB for all of them.

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
| `o200k_base` | OpenAI | `vocab-o200k_base` | 815 KB | GPT-4o, GPT-4.1, GPT-4.5, GPT-5–5.6 (incl. Sol/Terra/Luna), GPT-6.x (assumed; unpublished), o1, o3, o4-mini |
| `o200k_harmony` | OpenAI | `vocab-o200k_base` | — | gpt-oss (harmony chat format) |
| `cl100k_base` | OpenAI | `vocab-cl100k_base` | 373 KB | GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, text-embedding-*, davinci-002, babbage-002 |
| `p50k_base` | OpenAI | `vocab-p50k_base` | 55 B + base | text-davinci-002/003, code-davinci-*, code-cushman-* |
| `p50k_edit` | OpenAI | `vocab-p50k_base` | — | text-davinci-edit-*, code-davinci-edit-* |
| `r50k_base` | OpenAI | `vocab-r50k_base` | 182 KB | GPT-3 era: davinci, curie, babbage, ada |
| `gpt2` | OpenAI | `vocab-r50k_base` | — | GPT-2 (alias for `r50k_base`) |
| `llama3` | Meta | `vocab-llama3` | 111 KB + base | Llama 3, 3.1, 3.2, 3.3 |
| `llama4` | Meta | `vocab-llama4` | 816 KB | Llama 4 Scout / Maverick, Muse |
| `deepseek_v3` | DeepSeek | `vocab-deepseek_v3` | 514 KB | DeepSeek V3, R1 |
| `deepseek_v4` | DeepSeek | `vocab-deepseek_v3` | — | DeepSeek V4 Pro / Flash, V4.1 Flash (V3 vocab + V4 special tokens) |
| `qwen2` | Alibaba | `vocab-qwen2` | 564 KB | Qwen 2.5, Qwen 3 |
| `qwen3_5` | Alibaba | `vocab-qwen3_5` | 989 KB | Qwen 3.5 – 3.8, Qwen 4 (assumed; unpublished) |
| `mistral_v3` | Mistral | `vocab-mistral_v3` | 525 KB | Mistral, Mixtral, Large 3, Small 4 (Tekken tokenizer) |
| `mistral_v15` | Mistral | `vocab-mistral_v15` | 853 KB | Mistral Large 4 (Tekken v15) |
| `kimi_k2` | Moonshot | `vocab-kimi_k2` | 659 KB | Kimi K2 / K2.5 / K2.6 |
| `kimi_k3` | Moonshot | `vocab-kimi_k2` | — | Kimi K3 / K3.1 (K2 vocab + K3 special tokens) |
| `glm4` | Zhipu | `vocab-glm4` | 578 KB | GLM-4.5 / 4.6 / 4.7 |
| `glm5` | Zhipu | `vocab-glm5` | 6 KB + base | GLM-5 / 5.1 / 5.2 / 5.3 |
| `minimax_m2` | MiniMax | `vocab-minimax_m2` | 822 KB | MiniMax M2 / M2.1 / M2.5 / M2.7 / M3 / M3.1 |

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
let model = pricing::get_model("claude-opus-5.5").unwrap();
let cost = model.estimate_cost_with_cache(500_000, 500_000, 200_000);

// cache writes are priced per cache lifetime, here 1 hour; premium-speed and
// background windows have their own estimators
assert_eq!(model.cache_write_per_1m(60), Some(8.0));
let fast = pricing::get_model("gpt-6-sol").unwrap().estimate_priority_cost(1_000, 1_000);
let background = pricing::get_model("gpt-6-luna").unwrap().estimate_background_cost(1_000, 1_000, 8);

// list all models for a provider
let models = pricing::models_by_provider(pricing::Provider::DeepSeek);

// ids as the provider APIs spell them — dashes where this table uses a dot,
// release-date suffixes, Bedrock and Vertex decoration — resolve too
let r = pricing::resolve_model("us.anthropic.claude-opus-5").unwrap();
assert_eq!(r.model.id, "claude-opus-5");
assert!(matches!(r.matched, pricing::Match::Normalized { .. }));

// `estimate_cost` goes through the same resolution; `get_model` stays exact
assert!(pricing::estimate_cost("claude-haiku-4-5-20251001", 1_000, 1_000).is_some());
```

Supports 153 models across OpenAI, Anthropic, Google, Meta, DeepSeek, Alibaba,
Mistral, Moonshot, Zhipu, MiniMax and Voyage.

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

The code is licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

The embedded vocabularies are not covered by that: each stays under its publisher's license. [LICENSE-3RD-PARTY](LICENSE-3RD-PARTY) lists every vocabulary with its source and license, [`licenses/`](licenses/) holds the full texts, and [NOTICE](NOTICE) the attributions they require. Some of those licenses set conditions on use: the Llama licenses (products above 700 million monthly active users need a license from Meta), Mistral's license for `mistral_v15` (Model-as-a-Service businesses need a commercial license), display clauses for Kimi and MiniMax above a size threshold, and use-based restrictions for DeepSeek. A build without a vocabulary's `vocab-*` feature does not contain that vocabulary.

Built with Llama.
