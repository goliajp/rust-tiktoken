# @goliapkg/tiktoken-wasm

[![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm?style=flat-square&logo=npm)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)
[![License](https://img.shields.io/npm/l/@goliapkg/tiktoken-wasm?style=flat-square)](#许可证)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

这个包为 [tiktoken](https://crates.io/crates/tiktoken) BPE 分词器提供 WebAssembly 绑定，可直接在浏览器或 Node.js 中使用多家厂商的分词器，速度接近原生实现。

## 安装

```bash
npm install @goliapkg/tiktoken-wasm
```

模块使用 WebAssembly SIMD（`simd128`）。浏览器最低版本为 Chrome 91、Firefox 89 或 Safari 16.4，旧版本无法编译该模块。

## 从源码构建

```bash
# 需要 wasm-pack：cargo install wasm-pack
cd tiktoken-wasm
wasm-pack build --target web --release --scope goliapkg
```

构建后，`pkg/` 目录中包含一个可直接发布到 npm 的完整包。包内文件如下。

- `tiktoken_wasm.js` 是 ES 模块，内含 wasm 加载器
- `tiktoken_wasm_bg.wasm` 是编译后的 wasm 二进制文件，包含全部 20 份词表，大小为 8.65 MB，gzip 压缩后为 8.01 MB
- `tiktoken_wasm.d.ts` 提供 TypeScript 类型声明
- `snippets/` 包含一小段 JavaScript 代码，用于将文本和 token ID 写入 wasm 内存或从中读出

从源码构建时，可用 `--no-default-features --features vocab-o200k_base` 选择词表，也可以选用 `tiktoken` 转发的其他 `vocab-*` feature 或厂商分组 feature。只选 `o200k_base` 时，wasm 文件为 1.64 MB，gzip 压缩后为 1.00 MB。

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
// ["cl100k_base", "o200k_base", ..., "minimax_m2"]

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

// 按模型名 —— OpenAI、Meta、DeepSeek、Qwen、Mistral、Moonshot、智谱、MiniMax
const enc2 = encodingForModel('gpt-4o')
const encName = modelToEncoding('llama-4-scout')  // "llama4"

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

使用 Vite 时，在 `vite.config.ts` 中添加以下插件。

```typescript
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'

export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
})
```

使用 webpack 5 时，在 `webpack.config.js` 中启用 wasm 实验功能。

```javascript
module.exports = {
  experiments: {
    asyncWebAssembly: true,
  },
}
```

使用 Next.js 时，将以下配置添加到 `next.config.js`。

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

返回全部 20 种可用编码的名称。

### `getEncoding(name: string): Encoding`

按编码名称获取分词器。以下列出各编码适用的模型。

- `cl100k_base`：GPT-4、GPT-3.5-turbo
- `o200k_base`：GPT-4o、GPT-4.1、GPT-5〜5.6、GPT-6（尚未确认）、o1、o3、o4-mini
- `o200k_harmony`：gpt-oss（harmony 对话格式）
- `p50k_base`：text-davinci-002/003
- `p50k_edit`：text-davinci-edit
- `r50k_base`：GPT-3（davinci、curie 等）
- `gpt2`：GPT-2（`r50k_base` 的别名）
- `llama3`：Meta Llama 3.x
- `llama4`：Meta Llama 4、Muse
- `deepseek_v3`：DeepSeek V3/R1
- `qwen2`：Qwen 2/2.5/3
- `qwen3_5`：Qwen 3.5〜3.8、Qwen 4（尚未确认）
- `mistral_v3`：Mistral/Codestral/Pixtral（Tekken）
- `mistral_v15`：Mistral Large 4（Tekken v15）
- `deepseek_v4`：DeepSeek V4 Pro/Flash、V4.1 Flash
- `kimi_k2`：Kimi K2/K2.5/K2.6
- `kimi_k3`：Kimi K3/K3.1
- `glm4`：GLM-4.5/4.6/4.7
- `glm5`：GLM-5/5.1/5.2/5.3
- `minimax_m2`：MiniMax M2、M3 系列

### `encodingForModel(model: string): Encoding`

按模型名称获取分词器，例如 `gpt-4o`、`llama-4-scout`、`deepseek-r1` 或 `qwen3-max`。

### `modelToEncoding(model: string): string | null`

查询模型对应的编码名称，无需加载编码。

### `Encoding`

| 方法 / 属性 | 类型 | 说明 |
|-------------------|------|-------------|
| `encode(text)` | `Uint32Array` | 将文本编码为 token ID |
| `encodeWithSpecialTokens(text)` | `Uint32Array` | 编码时识别特殊 token |
| `decode(tokens)` | `string` | 将 token ID 解码为文本 |
| `count(text)` | `number` | 统计 token 数量（比 `encode().length` 快）|
| `countWithSpecialTokens(text)` | `number` | 统计 token 数量，计数时识别特殊 token |
| `name` | `string` | 编码名（getter）|
| `vocabSize` | `number` | 词表中的普通 token 数量 |
| `numSpecialTokens` | `number` | 特殊 token 数 |
| `free()` | `void` | 释放 WASM 内存 |

### `estimateCost(modelId, inputTokens, outputTokens): number`

以美元估算 API 调用费用，支持 11 家厂商的 153 个模型。

### `getModelInfo(modelId): ModelInfo`

返回模型元数据，并提供完整的 TypeScript 类型定义。

模型 ID 的解析规则与 `estimateCost` 相同。传入厂商 API 使用的名称（如 `claude-haiku-4-5` 或 `anthropic.claude-opus-5`）即可查到对应模型。返回的 `id` 使用模型表中记录的名称。

### `allModels(): ModelInfo[]`

列出支持的全部 153 个模型及其价格。

### `modelsByProvider(provider): ModelInfo[]`

按厂商过滤模型：`"OpenAI"`、`"Anthropic"`、`"Google"`、`"Meta"`、`"DeepSeek"`、`"Alibaba"`、`"Mistral"`、`"Moonshot"`、`"Zhipu"`、`"MiniMax"`、`"Voyage"`。

### `ModelInfo`

| 字段 | 类型 | 说明 |
|----------|------|-------------|
| `id` | `string` | 模型标识 |
| `provider` | `string` | 厂商名 |
| `inputPer1m` | `number` | 每百万个输入 token 的费用（美元）|
| `outputPer1m` | `number` | 每百万个输出 token 的费用（美元）|
| `cachedInputPer1m` | `number \| undefined` | 每百万个缓存输入 token 的费用 |
| `contextWindow` | `number` | 上下文窗口上限（以 token 为单位）|
| `maxOutput` | `number` | 最多可输出的 token 数量 |

## 支持的模型（含价格）

`estimateCost`、`getModelInfo` 和 `allModels` 支持 11 家厂商的 153 个模型，价格取自 2026 年 11 月。下表按厂商分组，每组先列出较新的模型。

| 厂商 | 模型数 | 最新条目 |
|----------|-------:|----------------|
| OpenAI | 41 | gpt-6.2-astra/sol、gpt-6.1-sol/terra、gpt-6-astra/sol/luna、gpt-5.6-sol/terra/luna、gpt-5.5(-pro)、gpt-5.4(-mini/nano/pro)、gpt-5.2(-pro)、gpt-5.1、gpt-5(-mini/nano/pro)、gpt-4.1、gpt-4o、o1/o3/o4-mini、… |
| Anthropic | 24 | claude-fable-5.2/5.1/5、claude-mythos-5.2/5.1/5、claude-opus-5.5/5、claude-sonnet-5.5/5、claude-haiku-5.5、claude-opus-4.8/4.7/4.6、claude-haiku-4.5、… |
| Google | 19 | gemini-4-neon/hydrogen/carbon/argon、gemini-3.8-flash、gemini-3.1-pro-preview、gemini-3.6/3.5-flash、gemini-embedding-2、gemini-2.5-pro/flash、… |
| Mistral | 15 | mistral-large-4、mistral-large/medium/small、codestral、devstral、ministral、mistral-embed、codestral-embed、pixtral-large、… |
| Alibaba | 13 | qwen4-max、qwen3.8-max/flash/27b/2.4t-a95b、qwen3.5-plus、qwen3-max/plus/coder、qwen2.5-72b、… |
| Meta | 8 | muse-spark-open、muse-glimmer-30b、llama-4-scout/maverick、llama-3.3-70b、llama-3.1-405b/70b/8b |
| Zhipu | 8 | glm-5.3(-flashx/-flash)、glm-5.2、glm-5、glm-4.7、glm-4.5(-air) |
| DeepSeek | 5 | deepseek-flash、deepseek-v4-pro、deepseek-v4-flash、deepseek-v3、deepseek-r1 |
| MiniMax | 6 | minimax-m3、minimax-m3.1-flash、minimax-m2.7/m2.5/m2.1/m2 |
| Moonshot | 6 | kimi-k3.1-pro(-highspeed)、kimi-k3、kimi-k2.7-code、kimi-k2.6、kimi-k2.5 |
| Voyage | 8 | voyage-4-large/4/4-lite/code-4、voyage-3-large、voyage-3.5(-lite)、voyage-code-3（嵌入模型）|

模型列表以代码为准，可在运行时调用 `allModels()` 获取，也可查阅 [docs.rs](https://docs.rs/tiktoken/latest/tiktoken/pricing/index.html)。Anthropic 和 Google 没有可供核对的公开分词器，本包仅提供这两家的价格数据。另有八家厂商同时支持分词。

## 注意事项

### 初始化

调用其他 API 前，先执行一次 `await init()` 以加载并编译 wasm 模块。后续初始化调用不会执行任何操作。

### 内存管理

`Encoding` 实例引用全局缓存的数据，自身开销很小。调用 `.free()` 会释放 JavaScript 包装对象，底层编码数据仍保留在缓存中，可供后续实例复用。短时间运行的脚本可以省略 `.free()`；长期运行的应用应在用完实例后调用该方法。

## 示例

[`examples/react-app`](../tiktoken/examples/react-app/) 提供了完整的 Vite + React 示例应用。

<!-- ECOSYSTEM BEGIN (generated — edit ecosystem.toml, not this block) -->

## 生态系统

[tiktoken](https://crates.io/crates/tiktoken) · **@goliapkg/tiktoken-wasm** · [instructors](https://crates.io/crates/instructors) · [chunkedrs](https://crates.io/crates/chunkedrs) · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## 许可证

代码采用双重许可，可任选 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE)。

内嵌词表不在上述许可范围内，各自沿用发布方的许可证。[LICENSE-3RD-PARTY](LICENSE-3RD-PARTY) 列出了每份词表的来源和许可证，[`licenses/`](licenses/) 收录了许可证全文，[NOTICE](NOTICE) 列出了各许可证要求的署名。

部分词表有额外的使用条件。根据 Llama 许可证，月活跃用户超过 7 亿的产品须向 Meta 取得许可。Mistral 要求提供模型即服务（Model-as-a-Service）的企业取得 `mistral_v15` 的商业许可。Kimi 和 MiniMax 的许可证对超过一定规模的使用者有展示方面的要求。DeepSeek 的许可证对用途有限制。构建时关闭某份词表的 `vocab-*` feature，即可从产物中排除该词表。

Built with Llama.
