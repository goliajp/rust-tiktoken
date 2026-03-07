# @goliapkg/tiktoken-wasm

[![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm?style=flat-square&logo=npm)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)
[![License](https://img.shields.io/npm/l/@goliapkg/tiktoken-wasm?style=flat-square)](../LICENSE)

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
- `tiktoken_wasm_bg.wasm` — compiled WASM binary (~7 MB, ~3 MB gzipped)
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
// ["cl100k_base", "o200k_base", ..., "mistral_v3"]

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
const encName = modelToEncoding('llama-4-scout')  // "llama3"

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

List all available encoding names (9 encodings).

### `getEncoding(name: string): Encoding`

Get a tokenizer by encoding name. Supported:
- `cl100k_base` — GPT-4, GPT-3.5-turbo
- `o200k_base` — GPT-4o, GPT-4.1, o1, o3
- `p50k_base` — text-davinci-002/003
- `p50k_edit` — text-davinci-edit
- `r50k_base` — GPT-3 (davinci, curie, etc.)
- `llama3` — Meta Llama 3/4
- `deepseek_v3` — DeepSeek V3/R1
- `qwen2` — Qwen 2/2.5/3
- `mistral_v3` — Mistral/Codestral/Pixtral

### `encodingForModel(model: string): Encoding`

Get a tokenizer by model name (e.g. `gpt-4o`, `llama-4-scout`, `deepseek-r1`, `qwen3-235b`).

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

Estimate API cost in USD. Supports 57 models across 7 providers.

### `getModelInfo(modelId): ModelInfo`

Get model metadata with full TypeScript typing.

### `allModels(): ModelInfo[]`

List all 57 supported models with pricing info.

### `modelsByProvider(provider): ModelInfo[]`

Filter models by provider: `"OpenAI"`, `"Anthropic"`, `"Google"`, `"Meta"`, `"DeepSeek"`, `"Alibaba"`, `"Mistral"`.

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

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, o3, o3-pro, o3-mini, o4-mini, o1, gpt-4-turbo, gpt-4, gpt-3.5-turbo, embeddings |
| Anthropic | claude-opus-4, claude-sonnet-4, claude-4.5-sonnet, claude-4.5-haiku, claude-4.6-sonnet, claude-4.6-opus, claude-4.6-haiku, claude-3.5-haiku, claude-3.5-sonnet, claude-3-opus, claude-3-haiku |
| Google | gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash |
| Meta | llama-4-scout, llama-4-maverick, llama-3.3-70b, llama-3.1-405b, llama-3.1-70b, llama-3.1-8b |
| DeepSeek | deepseek-r1, deepseek-v3, deepseek-chat |
| Qwen | qwen3-235b, qwen3-32b, qwen3-30b-a3b, qwen3-14b, qwen-2.5-72b, qwen-2.5-coder-32b, qwen-turbo |
| Mistral | mistral-large, mistral-medium, mistral-small, codestral, pixtral-large |

## Demo

See [`examples/react-app`](../examples/react-app/) for a complete Vite + React demo application.

## License

[MIT](../LICENSE)
