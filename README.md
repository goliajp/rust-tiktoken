# rust-tiktoken

[![tiktoken on crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust&label=tiktoken)](https://crates.io/crates/tiktoken)
[![tiktoken-wasm on npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm?style=flat-square&logo=npm&label=tiktoken-wasm)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-tiktoken/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-tiktoken/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#license)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) · **[tiktoken.golia.jp](https://tiktoken.golia.jp)** — live in-browser playground

> **Upgrading from 3.5.x?** 3.6.0, 3.8.0 and 5.0.0 corrected token ids that diverged from the reference tokenizers — recompute cached counts and stored id sequences. Details: [CHANGELOG](tiktoken/CHANGELOG.md#upgrading-from-35x--token-id-changes).
>
> **Upgrading to 4.0?** Token ids and the API are unchanged. The one break: vocabularies now live behind default-on `vocab-*` features, so a dependency written as `default-features = false` must name the ones it uses.
>
> **Upgrading to 5.2?** No API change. Texts of 32 KiB and more are now encoded on several threads by default (`tiktoken::set_parallelism(1)` or `TIKTOKEN_THREADS=1` turns that off); the wasm package needs WebAssembly SIMD (Chrome 91, Firefox 89, Safari 16.4); and 336 code points that are digits only in newer Unicode versions split differently on `mistral_v15` and `deepseek_v3` / `deepseek_v4`. Details: [CHANGELOG](tiktoken/CHANGELOG.md#520---2026-11-23).
>
> **Upgrading to 5.0?** Two breaks. The pricing structs and `VisionPricing` are `#[non_exhaustive]`, so a `match` needs a `_` arm and the structs can no longer be built with literals. And Llama 4, Qwen 3.5–3.8, Mistral Large 4 and `deepseek-flash` now resolve to the vocabularies those models really use, so their token counts change. 5.1 moves the vocabulary data into per-vendor `tiktoken-vocab-*` crates without changing anything else.

The fastest Rust BPE tokenizer, plus its WebAssembly bindings. Drop-in compatible with OpenAI [tiktoken](https://github.com/openai/tiktoken) and the mainstream open models (Llama 3, DeepSeek, Qwen, Mistral, Kimi, GLM, MiniMax). Hand-written scanners for ASCII **and CJK**, a key-size-layered vocabulary and whole-piece memoisation make it **22–83x faster than tiktoken-rs** and **1.5–12x faster than fastokens** natively, and **5.8–7.0x faster than gpt-tokenizer in the browser** — CJK prose included.

## Crates in this workspace

| Path | Crate / Package | Description | Version |
|:-----|:----------------|:------------|:--------|
| [`tiktoken/`](tiktoken/) | [`tiktoken`](https://crates.io/crates/tiktoken) | Rust BPE tokenizer — 20 encodings, 153 models, multi-provider pricing | [![crates.io](https://img.shields.io/crates/v/tiktoken.svg?style=flat-square)](https://crates.io/crates/tiktoken) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`tiktoken-wasm`](https://crates.io/crates/tiktoken-wasm) (Rust) | WASM binding crate for the above | [![crates.io](https://img.shields.io/crates/v/tiktoken-wasm.svg?style=flat-square)](https://crates.io/crates/tiktoken-wasm) |
| [`tiktoken-wasm/`](tiktoken-wasm/) | [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) (npm) | Same, published to npm via `wasm-pack` | [![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm.svg?style=flat-square)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) |

> The two crates live in one workspace and are **versioned in lockstep** — every release bumps and publishes both at the same version number.

## Highlights

- **No regex at runtime** — every encoding's split pattern is reproduced by a table-driven scanner built on generated Unicode tables, and the cl100k and o200k families decide piece starts 64 bytes at a time with SIMD (NEON, SSE2, wasm simd128). The `regex` crate is only the reference the scanners are tested against.
- **20 encodings · 153 models · 11 providers** — OpenAI (GPT-4/4o/4.1/4.5, GPT-5.x, GPT-6.x, o1/o3/o4-mini, gpt-oss), Llama 3/4, DeepSeek V3/V4, Qwen 2–3.8, Mistral (incl. Large 4), Kimi K2/K3, GLM-4/5, MiniMax M2/M3; plus USD cost estimation (Anthropic & Google included for pricing).
- **Repeated texts** — `enc.memo(max_bytes)` gives an opt-in `TextMemo` that answers a text it has seen before from memory (3.5 µs for a 200 KB book instead of 110 µs); `encode` itself keeps nothing.
- **Lean & portable** — all 20 vocabularies embedded in 7.8 MB and opt-out per vocabulary (a cl100k-only build carries 373 KB, and cargo only downloads the vendors you enable), key-size-layered vocabulary with whole-piece memoisation, hybrid linear/heap BPE merge, long texts encoded on several threads by a built-in std-only pool, allocation-free `count()`, pure Rust with zero C dependencies (runtime dependencies: `ruzstd` and the vocabulary data crates), and a self-contained wasm build.

Full API, supported-model tables, and benchmarks live in the per-crate READMEs: **[`tiktoken/`](tiktoken/README.md)** · **[`tiktoken-wasm/`](tiktoken-wasm/README.md)**.

## Quick start

### Rust

```toml
[dependencies]
tiktoken = "5"
```

```rust
// by encoding name
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode("hello world");
assert_eq!(enc.decode_to_string(&tokens).unwrap(), "hello world");

// count without allocating a token vector
let n = enc.count("The quick brown fox.");

// or resolve by model name
let enc = tiktoken::encoding_for_model("gpt-4o").unwrap();
```

### WebAssembly (browser / Node.js)

```bash
npm install @goliapkg/tiktoken-wasm
```

```js
import init, { getEncoding } from '@goliapkg/tiktoken-wasm'
await init()
const enc = getEncoding('o200k_base')
const tokens = enc.encode('hello world')
```

## Performance

On an Apple M4 Pro Mac mini, `encode` is **22–83x faster than tiktoken-rs** and **43–92x faster than Python tiktoken** on one thread: 22–83x on ASCII, 50–82x on Chinese and Japanese prose, 36x on an adversarial no-repeat CJK corpus. It is 1.5–12x faster than fastokens on one thread and 3x with default threads on a 200 KB book. In the browser (wasm) it is 5.8–7.0x faster than gpt-tokenizer and 1.1–2.5x faster than tiktoken-fast. Full tables and methodology: [`tiktoken/README.md#performance`](tiktoken/README.md#performance).

## Build

```bash
cargo test -p tiktoken
cargo fmt --all --check
cargo clippy --workspace --lib -- -D warnings

# WASM (requires wasm-pack: cargo install wasm-pack)
cd tiktoken-wasm
wasm-pack build --target web --release --scope goliapkg
```

## Release

`tiktoken` and `tiktoken-wasm` are versioned in lockstep and released together via git-flow (no PRs):

```bash
# release gate: the full property sweep (~35 minutes; CI runs a smoke sample)
PROPTEST_CASES=25000 cargo test --workspace --all-targets

git flow release start X.Y.Z
# bump versions to X.Y.Z: tiktoken/Cargo.toml, tiktoken-wasm/Cargo.toml
# (and its tiktoken path-dep); finalize both CHANGELOGs
git flow release finish X.Y.Z                       # merge → master, tag vX.Y.Z, back-merge develop
git tag -a tiktoken-wasm-vX.Y.Z vX.Y.Z^{commit} -m "tiktoken-wasm X.Y.Z"
git push origin master develop vX.Y.Z tiktoken-wasm-vX.Y.Z
# tag `v*` publishes the tiktoken crate; `tiktoken-wasm-v*` publishes the wasm crate + npm
```

The 7 pre-tokenizer property tests (each run on every pattern family), an
exhaustive sweep of short strings and 12 round-trip property tests pin the
scanners against the `regex` crate. `PROPTEST_CASES` sets how much new ground
each run covers; the seeds in `proptest-regressions/` replay before any random
case at every setting, so past failures are always re-checked.

## License

The code is licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

The embedded vocabularies are not covered by that: each stays under its publisher's license. [LICENSE-3RD-PARTY](tiktoken/LICENSE-3RD-PARTY) lists every vocabulary with its source and license, [`licenses/`](tiktoken/licenses/) holds the full texts, and [NOTICE](tiktoken/NOTICE) the attributions they require. Some of those licenses set conditions on use: the Llama licenses (products above 700 million monthly active users need a license from Meta), Mistral's license for `mistral_v15` (Model-as-a-Service businesses need a commercial license), display clauses for Kimi and MiniMax above a size threshold, and use-based restrictions for DeepSeek. A build without a vocabulary's `vocab-*` feature does not contain that vocabulary.

Built with Llama.
