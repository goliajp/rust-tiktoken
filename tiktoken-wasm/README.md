# @goliapkg/tokenrs-wasm

[![npm](https://img.shields.io/npm/v/@goliapkg/tokenrs-wasm?style=flat-square&logo=npm)](https://www.npmjs.com/package/@goliapkg/tokenrs-wasm)
[![License](https://img.shields.io/npm/l/@goliapkg/tokenrs-wasm?style=flat-square)](../LICENSE)

WebAssembly bindings for the [tiktoken](https://crates.io/crates/tiktoken) BPE tokenizer — run OpenAI-compatible tokenization directly in the browser or Node.js with near-native performance.

## Install

```bash
npm install @goliapkg/tokenrs-wasm
```

## Build from source

```bash
# requires wasm-pack: cargo install wasm-pack
cd tiktoken-wasm
wasm-pack build --target web --release
```

Output is in `pkg/` — a complete npm-ready package containing:
- `tiktoken_wasm.js` — ES module with WASM loader
- `tiktoken_wasm_bg.wasm` — compiled WASM binary (~3 MB)
- `tiktoken_wasm.d.ts` — TypeScript type definitions

## Usage

### ES Module (Browser / Vite / webpack)

```typescript
import init, {
  getEncoding,
  encodingForModel,
  estimateCost,
  getModelInfo,
  type Encoding,
} from '@goliapkg/tokenrs-wasm'

// initialize WASM module (required once, before any other calls)
await init()

// encode / decode
const enc: Encoding = getEncoding('cl100k_base')
const tokens: Uint32Array = enc.encode('hello world')
const text: string = enc.decode(tokens)   // "hello world"
const count: number = enc.count('hello world')  // 2

// by model name
const enc2 = encodingForModel('gpt-4o')

// cost estimation (USD)
const cost: number = estimateCost('gpt-4o', 1000, 500)

// model metadata
const info = getModelInfo('claude-opus-4')
// { id, provider, input_per_1m, output_per_1m, cached_input_per_1m, context_window, max_output }

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

### `getEncoding(name: string): Encoding`

Get a tokenizer by encoding name. Supported: `cl100k_base`, `o200k_base`, `p50k_base`, `p50k_edit`, `r50k_base`.

### `encodingForModel(model: string): Encoding`

Get a tokenizer by OpenAI model name (e.g. `gpt-4o`, `o3-mini`, `gpt-3.5-turbo`).

### `Encoding`

| Method | Returns | Description |
|--------|---------|-------------|
| `encode(text)` | `Uint32Array` | Encode text to token ids |
| `encodeWithSpecialTokens(text)` | `Uint32Array` | Encode with special token recognition |
| `decode(tokens)` | `string` | Decode token ids to text |
| `count(text)` | `number` | Count tokens (faster than `encode().length`) |
| `name` | `string` | Encoding name (getter) |
| `free()` | `void` | Release WASM memory |

### `estimateCost(modelId, inputTokens, outputTokens): number`

Estimate API cost in USD. Supports OpenAI, Anthropic Claude, and Google Gemini models.

### `getModelInfo(modelId): object`

Get model metadata: pricing, context window, max output tokens.

## Supported Models (pricing)

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o, gpt-4o-mini, o1, o3, o4-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, embeddings |
| Anthropic | claude-opus-4, claude-sonnet-4, claude-3.5-haiku, claude-3.5-sonnet, claude-3-opus, claude-3-haiku |
| Google | gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash |

## Demo

See [`examples/react-app`](../examples/react-app/) for a complete Vite + React demo application.

## License

[MIT](../LICENSE)
