# @goliapkg/tiktoken-wasm

[![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm?style=flat-square&logo=npm)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)
[![License](https://img.shields.io/npm/l/@goliapkg/tiktoken-wasm?style=flat-square)](#license)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

WebAssembly bindings for the [tiktoken](https://crates.io/crates/tiktoken) BPE tokenizer — run multi-provider tokenization directly in the browser or Node.js with near-native performance.

## Install

```bash
npm install @goliapkg/tiktoken-wasm
```

## Build from source

```bash
# requires wasm-pack: cargo install wasm-pack
cd tiktoken-wasm
wasm-pack build --target web --release
```

Output is in `pkg/` — a complete npm-ready package containing:
- `tiktoken_wasm.js` — ES module with WASM loader
- `tiktoken_wasm_bg.wasm` — compiled WASM binary (8.96 MB, 8.22 MB gzipped, all 20 vocabularies)

Building it yourself, `--no-default-features --features vocab-o200k_base` (or
any other `vocab-*` / vendor group forwarded from `tiktoken`) drops that to
1.92 MB / 1.20 MB gzipped.
- `tiktoken_wasm.d.ts` — TypeScript type definitions

## Usage

### ES Module (Browser / Vite / webpack)

```typescript
import init, {
  getEncoding,
  encodingForModel,
  listEncodings,
  modelToEncoding,
  estimateCost,
  getModelInfo,
  allModels,
  modelsByProvider,
  type Encoding,
  type ModelInfo,
} from '@goliapkg/tiktoken-wasm'

// initialize WASM module (required once, before any other calls)
await init()

// discover available encodings
const names: string[] = listEncodings()
// ["cl100k_base", "o200k_base", ..., "minimax_m2"]

// encode / decode
const enc: Encoding = getEncoding('cl100k_base')
const tokens: Uint32Array = enc.encode('hello world')
const text: string = enc.decode(tokens)   // "hello world"
const count: number = enc.count('hello world')  // 2

// special token handling
const countST: number = enc.countWithSpecialTokens('hi<|endoftext|>bye')

// vocabulary info
console.log(enc.vocabSize)         // 100256
console.log(enc.numSpecialTokens)  // 5

// by model name — supports OpenAI, Meta, DeepSeek, Qwen, Mistral
const enc2 = encodingForModel('gpt-4o')
const encName = modelToEncoding('llama-4-scout')  // "llama4"

// cost estimation (USD)
const cost: number = estimateCost('gpt-4o', 1000, 500)

// model metadata (fully typed)
const info: ModelInfo = getModelInfo('claude-opus-4')
console.log(info.id, info.provider, info.inputPer1m, info.contextWindow)

// browse all models or filter by provider
const all: ModelInfo[] = allModels()
const openai: ModelInfo[] = modelsByProvider('OpenAI')

// free WASM memory when done
enc.free()
enc2.free()
```

### Bundler Configuration

**Vite** — add plugins to `vite.config.ts`:

```typescript
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'

export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
})
```

**webpack 5** — enable WASM experiments in `webpack.config.js`:

```javascript
module.exports = {
  experiments: {
    asyncWebAssembly: true,
  },
}
```

**Next.js** — add to `next.config.js`:

```javascript
module.exports = {
  webpack: (config) => {
    config.experiments = { ...config.experiments, asyncWebAssembly: true }
    return config
  },
}
```

## API Reference

### `listEncodings(): string[]`

List all available encoding names (17 encodings).

### `getEncoding(name: string): Encoding`

Get a tokenizer by encoding name. Supported:
- `cl100k_base` — GPT-4, GPT-3.5-turbo
- `o200k_base` — GPT-4o, GPT-4.1, GPT-5–5.6, GPT-6 (unconfirmed), o1, o3, o4-mini
- `o200k_harmony` — gpt-oss (harmony chat format)
- `p50k_base` — text-davinci-002/003
- `p50k_edit` — text-davinci-edit
- `r50k_base` — GPT-3 (davinci, curie, etc.)
- `gpt2` — GPT-2 (alias for `r50k_base`)
- `llama3` — Meta Llama 3.x
- `llama4` — Meta Llama 4, Muse
- `deepseek_v3` — DeepSeek V3/R1
- `qwen2` — Qwen 2/2.5/3
- `qwen3_5` — Qwen 3.5–3.8, Qwen 4 (unconfirmed)
- `mistral_v3` — Mistral/Codestral/Pixtral (Tekken)
- `mistral_v15` — Mistral Large 4 (Tekken v15)
- `deepseek_v4` — DeepSeek V4 Pro/Flash, V4.1 Flash
- `kimi_k2` — Kimi K2/K2.5/K2.6
- `kimi_k3` — Kimi K3/K3.1
- `glm4` — GLM-4.5/4.6/4.7
- `glm5` — GLM-5/5.1/5.2/5.3
- `minimax_m2` — MiniMax M2 and M3 families

### `encodingForModel(model: string): Encoding`

Get a tokenizer by model name (e.g. `gpt-4o`, `llama-4-scout`, `deepseek-r1`, `qwen3-max`).

### `modelToEncoding(model: string): string | null`

Map a model name to its encoding name without loading the encoding.

### `Encoding`

| Method / Property | Type | Description |
|-------------------|------|-------------|
| `encode(text)` | `Uint32Array` | Encode text to token ids |
| `encodeWithSpecialTokens(text)` | `Uint32Array` | Encode with special token recognition |
| `decode(tokens)` | `string` | Decode token ids to text |
| `count(text)` | `number` | Count tokens (faster than `encode().length`) |
| `countWithSpecialTokens(text)` | `number` | Count tokens with special token recognition |
| `name` | `string` | Encoding name (getter) |
| `vocabSize` | `number` | Number of regular tokens in vocabulary |
| `numSpecialTokens` | `number` | Number of special tokens |
| `free()` | `void` | Release WASM memory |

### `estimateCost(modelId, inputTokens, outputTokens): number`

Estimate API cost in USD. Supports 153 models across 11 providers.

### `getModelInfo(modelId): ModelInfo`

Get model metadata with full TypeScript typing.

The id is resolved the way `estimateCost` resolves it, so a model addressed as
its API spells it (`claude-haiku-4-5`, `anthropic.claude-opus-5`) returns that
model. The returned `id` is this table's own spelling.

### `allModels(): ModelInfo[]`

List all 153 supported models with pricing info.

### `modelsByProvider(provider): ModelInfo[]`

Filter models by provider: `"OpenAI"`, `"Anthropic"`, `"Google"`, `"Meta"`, `"DeepSeek"`, `"Alibaba"`, `"Mistral"`, `"Moonshot"`, `"Zhipu"`, `"MiniMax"`, `"Voyage"`.

### `ModelInfo`

| Property | Type | Description |
|----------|------|-------------|
| `id` | `string` | Model identifier |
| `provider` | `string` | Provider name |
| `inputPer1m` | `number` | Input cost per 1M tokens (USD) |
| `outputPer1m` | `number` | Output cost per 1M tokens (USD) |
| `cachedInputPer1m` | `number \| undefined` | Cached input cost per 1M tokens |
| `contextWindow` | `number` | Max context window (tokens) |
| `maxOutput` | `number` | Max output tokens |

## Supported Models (pricing)

`estimateCost` / `getModelInfo` / `allModels` cover **153 models across 11
providers** (2026-11 pricing). Per provider, newest first:

| Provider | Models | Latest entries |
|----------|-------:|----------------|
| OpenAI | 41 | gpt-6.2-astra/sol, gpt-6.1-sol/terra, gpt-6-astra/sol/luna, gpt-5.6-sol/terra/luna, gpt-5.5(-pro), gpt-5.4(-mini/nano/pro), gpt-5.2(-pro), gpt-5.1, gpt-5(-mini/nano/pro), gpt-4.1, gpt-4o, o1/o3/o4-mini, … |
| Anthropic | 24 | claude-fable-5.2/5.1/5, claude-mythos-5.2/5.1/5, claude-opus-5.5/5, claude-sonnet-5.5/5, claude-haiku-5.5, claude-opus-4.8/4.7/4.6, claude-haiku-4.5, … |
| Google | 19 | gemini-4-neon/hydrogen/carbon/argon, gemini-3.8-flash, gemini-3.1-pro-preview, gemini-3.6/3.5-flash, gemini-embedding-2, gemini-2.5-pro/flash, … |
| Mistral | 15 | mistral-large-4, mistral-large/medium/small, codestral, devstral, ministral, mistral-embed, codestral-embed, pixtral-large, … |
| Alibaba | 13 | qwen4-max, qwen3.8-max/flash/27b/2.4t-a95b, qwen3.5-plus, qwen3-max/plus/coder, qwen2.5-72b, … |
| Meta | 8 | muse-spark-open, muse-glimmer-30b, llama-4-scout/maverick, llama-3.3-70b, llama-3.1-405b/70b/8b |
| Zhipu | 8 | glm-5.3(-flashx/-flash), glm-5.2, glm-5, glm-4.7, glm-4.5(-air) |
| DeepSeek | 5 | deepseek-flash, deepseek-v4-pro, deepseek-v4-flash, deepseek-v3, deepseek-r1 |
| MiniMax | 6 | minimax-m3, minimax-m3.1-flash, minimax-m2.7/m2.5/m2.1/m2 |
| Moonshot | 6 | kimi-k3.1-pro(-highspeed), kimi-k3, kimi-k2.7-code, kimi-k2.6, kimi-k2.5 |
| Voyage | 8 | voyage-4-large/4/4-lite/code-4, voyage-3-large, voyage-3.5(-lite), voyage-code-3 (embeddings) |

The authoritative list is the code: call `allModels()` at runtime, or see
[docs.rs](https://docs.rs/tiktoken/latest/tiktoken/pricing/index.html).
Anthropic and Google models are pricing-only (no public tokenizer to match);
the other eight providers tokenize too.

## Notes

### Initialization

Call `await init()` once before any other API calls. This loads and compiles the WASM module. Subsequent calls are a no-op.

### Memory Management

`Encoding` instances hold references to globally cached data and are lightweight. Calling `.free()` releases the JS wrapper — the underlying encoding data remains cached for reuse. In short-lived scripts you can skip `.free()`; in long-running apps, call it when you're done with the instance.

## Demo

See [`examples/react-app`](../tiktoken/examples/react-app/) for a complete Vite + React demo application.

<!-- ECOSYSTEM BEGIN (generated — edit ecosystem.toml, not this block) -->

## Ecosystem

[tiktoken](https://crates.io/crates/tiktoken) · **@goliapkg/tiktoken-wasm** · [instructors](https://crates.io/crates/instructors) · [chunkedrs](https://crates.io/crates/chunkedrs) · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## License

The code is licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

The embedded vocabularies are not covered by that: each stays under its publisher's license. [LICENSE-3RD-PARTY](LICENSE-3RD-PARTY) lists every vocabulary with its source and license, [`licenses/`](licenses/) holds the full texts, and [NOTICE](NOTICE) the attributions they require. Some of those licenses set conditions on use: the Llama licenses (products above 700 million monthly active users need a license from Meta), Mistral's license for `mistral_v15` (Model-as-a-Service businesses need a commercial license), display clauses for Kimi and MiniMax above a size threshold, and use-based restrictions for DeepSeek. A build without a vocabulary's `vocab-*` feature does not contain that vocabulary.

Built with Llama.
