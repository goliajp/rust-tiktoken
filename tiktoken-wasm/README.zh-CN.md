# @goliapkg/tiktoken-wasm

[![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm?style=flat-square&logo=npm)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)
[![License](https://img.shields.io/npm/l/@goliapkg/tiktoken-wasm?style=flat-square)](LICENSE)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

[tiktoken](https://crates.io/crates/tiktoken) BPE 分词器的 WebAssembly 绑定 —— 在浏览器或 Node.js 中以接近原生性能直接跑多厂商分词。

## 安装

```bash
npm install @goliapkg/tiktoken-wasm
```

## 从源码构建

```bash
# 需要 wasm-pack：cargo install wasm-pack
cd tiktoken-wasm
wasm-pack build --target web --release --scope goliapkg
```

输出在 `pkg/` 目录下，是一份可直接发 npm 的完整包：

- `tiktoken_wasm.js` —— 带 WASM 加载器的 ES module
- `tiktoken_wasm_bg.wasm` —— 编译后的 WASM 二进制（~7 MB，gzip 后 ~3 MB）
- `tiktoken_wasm.d.ts` —— TypeScript 类型声明

## 使用

### ES Module（浏览器 / Vite / webpack）

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

// 初始化 WASM 模块（所有其它调用前必须调用一次）
await init()

// 列出可用编码
const names: string[] = listEncodings()
// ["cl100k_base", "o200k_base", ..., "mistral_v3"]

// 编码 / 解码
const enc: Encoding = getEncoding('cl100k_base')
const tokens: Uint32Array = enc.encode('hello world')
const text: string = enc.decode(tokens)   // "hello world"
const count: number = enc.count('hello world')  // 2

// 特殊 token 处理
const countST: number = enc.countWithSpecialTokens('hi<|endoftext|>bye')

// 词表信息
console.log(enc.vocabSize)         // 100256
console.log(enc.numSpecialTokens)  // 5

// 按模型名取编码 —— 支持 OpenAI、Meta、DeepSeek、Qwen、Mistral
const enc2 = encodingForModel('gpt-4o')
const encName = modelToEncoding('llama-4-scout')  // "llama3"

// 成本估算（美元）
const cost: number = estimateCost('gpt-4o', 1000, 500)

// 模型元数据（完整 TypeScript 类型）
const info: ModelInfo = getModelInfo('claude-opus-4')
console.log(info.id, info.provider, info.inputPer1m, info.contextWindow)

// 浏览全部模型或按厂商过滤
const all: ModelInfo[] = allModels()
const openai: ModelInfo[] = modelsByProvider('OpenAI')

// 用完释放 WASM 内存
enc.free()
enc2.free()
```

### 打包工具配置

**Vite** —— 在 `vite.config.ts` 里加插件：

```typescript
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'

export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
})
```

**webpack 5** —— 在 `webpack.config.js` 里开 WASM experiments：

```javascript
module.exports = {
  experiments: {
    asyncWebAssembly: true,
  },
}
```

**Next.js** —— 加到 `next.config.js`：

```javascript
module.exports = {
  webpack: (config) => {
    config.experiments = { ...config.experiments, asyncWebAssembly: true }
    return config
  },
}
```

## API 参考

### `listEncodings(): string[]`

列出全部可用编码名（共 9 套）。

### `getEncoding(name: string): Encoding`

按编码名取分词器。支持：

- `cl100k_base` —— GPT-4、GPT-3.5-turbo
- `o200k_base` —— GPT-4o、GPT-4.1、o1、o3
- `p50k_base` —— text-davinci-002/003
- `p50k_edit` —— text-davinci-edit
- `r50k_base` —— GPT-3（davinci、curie 等）
- `llama3` —— Meta Llama 3/4
- `deepseek_v3` —— DeepSeek V3/R1
- `qwen2` —— Qwen 2/2.5/3
- `mistral_v3` —— Mistral/Codestral/Pixtral

### `encodingForModel(model: string): Encoding`

按模型名取分词器（例：`gpt-4o`、`llama-4-scout`、`deepseek-r1`、`qwen3-max`）。

### `modelToEncoding(model: string): string | null`

把模型名映射到编码名，不加载编码本身。

### `Encoding`

| 方法 / 属性 | 类型 | 说明 |
|-------------------|------|-------------|
| `encode(text)` | `Uint32Array` | 文本 → token id |
| `encodeWithSpecialTokens(text)` | `Uint32Array` | 编码时识别特殊 token |
| `decode(tokens)` | `string` | token id → 文本 |
| `count(text)` | `number` | 只数 token 数（比 `encode().length` 快） |
| `countWithSpecialTokens(text)` | `number` | 识别特殊 token 的计数 |
| `name` | `string` | 编码名（getter） |
| `vocabSize` | `number` | 普通 token 词表大小 |
| `numSpecialTokens` | `number` | 特殊 token 数 |
| `free()` | `void` | 释放 WASM 内存 |

### `estimateCost(modelId, inputTokens, outputTokens): number`

按美元估算 API 成本。覆盖 7 家厂商 57 个模型。

### `getModelInfo(modelId): ModelInfo`

拿完整 TypeScript 类型的模型元数据。

### `allModels(): ModelInfo[]`

列出全部 57 个支持模型及其价格信息。

### `modelsByProvider(provider): ModelInfo[]`

按厂商过滤模型：`"OpenAI"`、`"Anthropic"`、`"Google"`、`"Meta"`、`"DeepSeek"`、`"Alibaba"`、`"Mistral"`。

### `ModelInfo`

| 字段 | 类型 | 说明 |
|----------|------|-------------|
| `id` | `string` | 模型标识 |
| `provider` | `string` | 厂商名 |
| `inputPer1m` | `number` | 每百万 token 输入成本（美元） |
| `outputPer1m` | `number` | 每百万 token 输出成本 |
| `cachedInputPer1m` | `number \| undefined` | 缓存命中的输入成本 |
| `contextWindow` | `number` | 最大上下文窗口（token） |
| `maxOutput` | `number` | 最大输出 token |

## 支持的模型（含价格）

| 厂商 | 模型 |
|----------|--------|
| OpenAI | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, o1, o1-mini, o1-pro, o3, o3-pro, o3-mini, o4-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002 |
| Anthropic | claude-opus-4.6, claude-sonnet-4.6, claude-haiku-4.5, claude-opus-4.5, claude-sonnet-4.5, claude-opus-4, claude-sonnet-4, claude-3.5-haiku, claude-3.5-sonnet, claude-3-opus, claude-3-haiku |
| Google | gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash, text-embedding-004 |
| Meta | llama-4-scout, llama-4-maverick, llama-3.1-405b, llama-3.1-70b, llama-3.1-8b, llama-3.3-70b |
| DeepSeek | deepseek-v3, deepseek-r1 |
| Alibaba | qwen3-max, qwen3-plus, qwen3-coder, qwen3-8b, qwen2.5-72b, qwen2.5-32b, qwen2.5-7b |
| Mistral | mistral-large, mistral-medium, mistral-small, mistral-nemo, codestral, pixtral-large, mixtral-8x7b |

## 注意事项

### 初始化

所有 API 调用前必须 `await init()` 一次。它会加载并编译 WASM 模块，之后再调用是 no-op。

### 内存管理

`Encoding` 实例持有全局缓存数据的引用，本身很轻。调 `.free()` 只释放 JS 包装器 —— 底层编码数据仍会被缓存复用。一次性脚本可以不调 `.free()`；长跑应用用完请调。

## Demo

完整 Vite + React 示例：[`examples/react-app`](../tiktoken/examples/react-app/)。

## 许可证

[MIT](LICENSE)
