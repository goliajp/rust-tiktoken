# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#license)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square&logo=rust)](Cargo.toml)
[![Downloads](https://img.shields.io/crates/d/tiktoken?style=flat-square)](https://crates.io/crates/tiktoken)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

The fastest Rust BPE tokenizer — 22–83x faster than tiktoken-rs natively (50–82x on Chinese and Japanese prose), 1.5–12x faster than fastokens, and 5.8–7.0x faster than gpt-tokenizer in the browser via wasm. Hand-written scanners cover ASCII and CJK alike, the vocabulary is layered by key size, and repeated pieces are memoised whole. Compatible with OpenAI [tiktoken](https://github.com/openai/tiktoken) and supports **all mainstream LLM tokenizers** — OpenAI, Llama 3, DeepSeek, Qwen, Mistral, Kimi, GLM, MiniMax.

## Features

- **Multi-provider**: 20 encodings across 8 vendors (OpenAI, Meta, DeepSeek, Alibaba, Mistral, Moonshot, Zhipu, MiniMax)
- **Fast**: table-driven pre-tokenizers for every encoding (no regex at runtime), SIMD piece-start detection for the cl100k and o200k families, key-size-layered vocabulary, whole-piece memoisation shared across threads, hybrid BPE merge
- **Parallel encoding**: texts of 32 KiB and more are encoded on several threads automatically, on a built-in std-only pool
- **Pricing**: cost estimation for 153 models across 11 providers
- **Compact**: all 20 vocabularies embedded in 7.8 MB, and opt-out per vocabulary — a cl100k-only build carries 373 KB of data, and cargo downloads only the vendors you enable
- **Zero-alloc counting**: `count()` path avoids token vector allocation

## Performance

Token outputs are asserted identical across implementations before anything is
timed; each figure is one full pass over the corpus, median of 9 rounds after
warmup. Corpora are byte-identical in every harness (`bench-compare/`,
`benches/bench_python.py`, `web/bench/`).

#### Native — Apple M4 Pro Mac mini, single thread, `encode`

`TIKTOKEN_THREADS=1 cargo run --release -p bench-compare`

| Corpus | Python tiktoken 0.14 | tiktoken-rs 0.12 | **tiktoken** | vs rs | vs Python |
|---|---|---|---|---|---|
| short (13 B) | 1.3 µs | 602 ns | **28 ns** | **22x** | **46x** |
| medium (900 B) | 23.2 µs | 19.8 µs | **431 ns** | **46x** | **54x** |
| English prose (45 KB) | 1,100 µs | 955 µs | **19.9 µs** | **48x** | **55x** |
| Chinese prose (4.3 KB) | 114 µs | 97.8 µs | **2.0 µs** | **50x** | **58x** |
| Japanese prose (4.6 KB) | 131 µs | 119 µs | **1.4 µs** | **82x** | **91x** |
| mixed CJK ×50 (4.5 KB) | 122 µs | 108 µs | **2.5 µs** | **43x** | **49x** |
| adversarial CJK, no repeats (3.9 KB) | 114 µs | 94.0 µs | **2.6 µs** | **36x** | **43x** |
| code (3.9 KB) | 179 µs | 163 µs | **1.9 µs** | **83x** | **92x** |

o200k_base runs 12–80x faster than tiktoken-rs on one thread. `count()` is up
to 36% faster than `encode` on short texts and about as fast on long ones — it
never allocates the id vector. With the default number of threads the 45 KB
text takes 6.2 µs.

#### Against the fastest alternatives — Apple M4 Pro Mac mini, one thread, cl100k_base, `encode`

Every implementation we found that could be faster, in one process on the same
inputs, after the same id-for-id check; a separate harness from `bench-compare`.
Figures are medians of 3 runs; `tiktoken` is 5.2.0 from crates.io.

| Corpus | **tiktoken** | [fastokens](https://crates.io/crates/fastokens) 0.3.6 | [HF tokenizers](https://crates.io/crates/tokenizers) 1.0-rc.5 | [toks](https://github.com/actual-computer/toks) 0.3.3 | [toktok-rs](https://crates.io/crates/toktok-rs) 0.5 | [magi-tokenizer](https://crates.io/crates/magi-tokenizer) 0.1 | [bpe-openai](https://crates.io/crates/bpe-openai) 0.3 |
|---|---|---|---|---|---|---|---|
| short (13 B) | **25 ns** | 61 ns | 112 ns | 42 ns | 32 ns | 27 ns | 147 ns |
| medium (900 B) | **431 ns** | 750 ns | 733 ns | 606 ns | 1.3 µs | 1.0 µs | 8.2 µs |
| English prose (45 KB) | **19.9 µs** | 35.6 µs | 32.3 µs | 29.9 µs | 61.2 µs | 49.5 µs | 405 µs |
| Chinese prose (4.3 KB) | **2.0 µs** | 8.0 µs | 4.3 µs | 12.9 µs | 10.1 µs | 12.2 µs | 37.7 µs |
| Japanese prose (4.6 KB) | **1.4 µs** | 17.2 µs | 4.1 µs | 18.8 µs | 15.8 µs | 22.8 µs | 41.1 µs |
| mixed CJK ×50 (4.5 KB) | **2.6 µs** | 3.9 µs | 5.8 µs | 17.8 µs | 12.7 µs | 18.9 µs | 40.0 µs |
| adversarial CJK, no repeats (3.9 KB) | **2.7 µs** | 5.3 µs | 5.4 µs | 17.2 µs | 15.3 µs | 19.4 µs | 56.1 µs |
| code (3.9 KB) | **1.9 µs** | 3.9 µs | 3.4 µs | 2.8 µs | 6.1 µs | 6.6 µs | 46.1 µs |
| English book (200 KB) | **111 µs** | 184 µs | 205 µs | 144 µs | 341 µs | 715 µs | 2,854 µs |
| Rust source (100 KB) | **56.1 µs** | 95.8 µs | 97.6 µs | 79.8 µs | 181 µs | 263 µs | 1,412 µs |

- **fastokens** — 1.5–12x slower on one thread. With default threads: 50.3 µs on the
  book (3.0x) and 36.3 µs on the Rust source (2.9x), against 16.7 µs and 12.5 µs here.
- **Short text**: magi-tokenizer is within 4–12% on the 13-byte string.
- **First encounter** (caches cleared, one thread): 1.7–6.2x faster than fastokens
  (book: 228 µs against 392 µs) and 1.17x or more ahead of every
  other library above 13 bytes; on 13 bytes several libraries tie at the timer's 42 ns tick.
  On the book with default threads the lead over fastokens is 8–20% when its workers
  keep their caches.
- **toks** is listed without its whole-text memo. With the memo on, encoding the *same*
  text again returns a stored result and is 3–20x faster than encoding it here. Against
  that mode, use [`CoreBpe::memo`](#repeated-texts): returning the stored ids it is 1.3–6x
  faster than toks' memo on this corpus set (cl100k and o200k); copying them into a
  buffer, as toks does, it ranges from 10% slower (45 KB) to 5x faster (13 bytes).
- Every other library tried — tiktoken-rs, tokensim, riptoken, kitoken, basetenkenizer,
  Hugging Face tokenizers 0.23 — was at least 3.4x slower on every row (Hugging Face 0.23
  and kitoken produced different ids on the Rust source, which is not timed). Gigatoken,
  callable only from Python, was 1.4–11x slower including the call overhead.

#### In the browser — Apple M4 Pro Mac mini, Chromium

`npm run bench` in `web/` — this crate compiled to wasm, against another wasm
tokenizer and the two mainstream JavaScript ones. All columns are `encode`.

| Corpus | tiktoken-fast 0.1 (wasm) | gpt-tokenizer 4.1 | js-tiktoken 1.0 | **tiktoken (wasm)** |
|---|---|---|---|---|
| Chinese prose (4.3 KB) | 6.4 µs | 31.9 µs | 6,625 µs | **4.8 µs** |
| Japanese prose (4.6 KB) | 6.6 µs | 29.2 µs | 12,825 µs | **4.2 µs** |
| mixed CJK ×50 (4.5 KB) | 7.0 µs | 40.3 µs | 3,975 µs | **6.3 µs** |
| adversarial CJK, no repeats (3.9 KB) | 6.7 µs | 42.2 µs | 2,756 µs | **6.1 µs** |
| English prose (45 KB) | 118 µs | 277 µs | 9,638 µs | **47.7 µs** |
| code (3.9 KB) | 8.9 µs | 30.4 µs | 1,120 µs | **5.1 µs** |

tiktoken-fast comes within about 10% on the two mixed-script corpora once it has seen
them; on a first encounter it is 1.8–6.6x slower. In Safari the lead over
gpt-tokenizer is 3.3–7.9x, over tiktoken-fast 1.5–2.2x.

The adversarial corpus never repeats a piece, which disables every
implementation's memoisation; the lead holds there too.

<details>
<summary>Why is it faster?</summary>

| | tiktoken | tiktoken-rs | Python tiktoken |
|---|---|---|---|
| Pre-tokenize | Table-driven scanners for every encoding, on Unicode tables generated from the patterns' own Unicode version; the cl100k and o200k families decide piece starts 64 bytes at a time with SIMD (NEON / SSE2 or AVX2 / wasm simd128) | always runs the regex | always runs the regex |
| Regex engine | none at runtime (`regex` is a dev-dependency, the reference the scanners are tested against) | `fancy-regex` (backtracking) | `regex` via PyO3 + FFI overhead |
| Vocab lookup | Layered by key size: packed keys of up to 15 bytes in one open-addressed table (two slots per cache line), hash-tagged arena slots above, direct-indexed tables for 1–2-byte keys | `HashMap<Vec<u8>>` (200k allocs) | Rust `HashMap` behind PyO3 |
| Repeated pieces | Merge results of non-vocabulary pieces are added to the shared lookup table, so every thread reuses them; long pieces have their own thread-local and shared caches | none | none |
| BPE merge | Hybrid: stack linear-scan (short pieces) + heap (long); short pieces that miss are merged several at a time by token id | O(n*m) linear scan | O(n*m) linear scan |
| `count()` without alloc | yes | no | no |

The scanners are pinned to the `regex` crate by property tests (hundreds of
thousands of random inputs per run) and an exhaustive sweep of short strings,
and every encoding is pinned to its vendor's tokenizer by 53,275 differential
fixture cases.

Benchmark source: [`benches/`](benches/), [`../bench-compare/`](../bench-compare/). Reproducible via `cargo bench` / `cargo run -p bench-compare`.

</details>

## Installation

```toml
[dependencies]
tiktoken = "5"
```

The runtime dependencies are `ruzstd` and the vocabulary data crates you
enable. The `parallel` feature of earlier versions still exists so that
manifests naming it keep building, but it does nothing: long texts are
encoded in parallel without it.

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
1.5 MB for OpenAI alone instead of 8.0 MB for all of them.

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

// one buffer for many calls: no allocation per call
let mut ids = Vec::new();
enc.encode_into("hello world", &mut ids);         // appends to `ids`

// one token's bytes, borrowed: lengths give byte offsets in the text
let hello: Option<&[u8]> = enc.token_bytes(ids[0]);
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

### Repeated texts

A `TextMemo` keeps whole texts and their ids. A text seen before is compared byte for
byte and its ids are returned as a slice; anything else is encoded and kept, which costs
at most about 4% extra the first time on texts of 3 KB and more (on 900 bytes, at most one
42 ns timer tick). Nothing is kept unless you create one.

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let mut memo = enc.memo(64 << 20); // at most about 64 MiB of texts and ids
let first = memo.encode("a long system prompt").to_vec();
assert_eq!(memo.encode("a long system prompt"), first); // answered from the memo
```

### Parallel encoding

`encode`, `count`, `encode_with_special_tokens` and `count_with_special_tokens`
cut a text of 32 KiB or more into chunks and encode them on a process-wide
pool of worker threads, created on the first such call. The output is
identical to single-threaded encoding.

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let text = "word ".repeat(20_000);
let tokens = enc.encode(&text); // 100 KB: encoded on several threads

// threads per call, the calling thread included; 1 = single-threaded,
// 0 = back to the default
tiktoken::set_parallelism(1);
assert_eq!(enc.encode(&text), tokens);
tiktoken::set_parallelism(0);
```

- The default thread count is the number of performance cores (all cores
  where the system does not tell them apart), or the `TIKTOKEN_THREADS`
  environment variable when it holds a positive integer.
- `TIKTOKEN_THREADS=1` or `set_parallelism(1)` turns it off: every call runs
  on the calling thread and no worker thread is ever started.
- One parallel call runs at a time. A call that finds the pool busy — for
  example because the application already encodes from several threads —
  runs on its own thread instead of waiting, so the process never runs more
  encoding threads than the configured number.
- `encode_parallel` is now the same as `encode`, kept for existing code; it
  no longer needs the `parallel` feature.
- On `wasm32` there are no threads, and every call runs single-threaded.

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

Available as [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) on npm. The package is built with WebAssembly SIMD and needs Chrome 91, Firefox 89 or Safari 16.4 and later.

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
wasm-pack build --target web --release --scope goliapkg
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
