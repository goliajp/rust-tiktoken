# @goliapkg/tiktoken-wasm

[![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm?style=flat-square&logo=npm)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)
[![License](https://img.shields.io/npm/l/@goliapkg/tiktoken-wasm?style=flat-square)](LICENSE)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

[tiktoken](https://crates.io/crates/tiktoken) BPE トークナイザの WebAssembly バインディング — ブラウザや Node.js 上で、ネイティブに迫る速度でマルチプロバイダのトークナイズを直接実行。

## インストール

```bash
npm install @goliapkg/tiktoken-wasm
```

## ソースからビルド

```bash
# wasm-pack が必要: cargo install wasm-pack
cd tiktoken-wasm
wasm-pack build --target web --release --scope goliapkg
```

出力は `pkg/` に入ります — そのまま npm に公開できる完全なパッケージ:

- `tiktoken_wasm.js` — WASM ローダ付きの ES モジュール
- `tiktoken_wasm_bg.wasm` — コンパイル済み WASM バイナリ（~7 MB、gzip 後 ~3 MB）
- `tiktoken_wasm.d.ts` — TypeScript 型定義

## 使い方

### ES Module（ブラウザ / Vite / webpack）

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

// WASM モジュールを初期化（他の呼び出しの前に一度だけ必要）
await init()

// 利用可能なエンコーディング一覧
const names: string[] = listEncodings()
// ["cl100k_base", "o200k_base", ..., "mistral_v3"]

// エンコード / デコード
const enc: Encoding = getEncoding('cl100k_base')
const tokens: Uint32Array = enc.encode('hello world')
const text: string = enc.decode(tokens)   // "hello world"
const count: number = enc.count('hello world')  // 2

// 特殊トークンの扱い
const countST: number = enc.countWithSpecialTokens('hi<|endoftext|>bye')

// 語彙情報
console.log(enc.vocabSize)         // 100256
console.log(enc.numSpecialTokens)  // 5

// モデル名から取得 — OpenAI、Meta、DeepSeek、Qwen、Mistral をサポート
const enc2 = encodingForModel('gpt-4o')
const encName = modelToEncoding('llama-4-scout')  // "llama3"

// コスト見積もり（USD）
const cost: number = estimateCost('gpt-4o', 1000, 500)

// モデルのメタデータ（完全に型付け）
const info: ModelInfo = getModelInfo('claude-opus-4')
console.log(info.id, info.provider, info.inputPer1m, info.contextWindow)

// 全モデルの閲覧、またはプロバイダで絞り込み
const all: ModelInfo[] = allModels()
const openai: ModelInfo[] = modelsByProvider('OpenAI')

// 使い終わったら WASM メモリを解放
enc.free()
enc2.free()
```

### バンドラの設定

**Vite** — `vite.config.ts` にプラグインを追加:

```typescript
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'

export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
})
```

**webpack 5** — `webpack.config.js` で WASM の experiments を有効化:

```javascript
module.exports = {
  experiments: {
    asyncWebAssembly: true,
  },
}
```

**Next.js** — `next.config.js` に追加:

```javascript
module.exports = {
  webpack: (config) => {
    config.experiments = { ...config.experiments, asyncWebAssembly: true }
    return config
  },
}
```

## API リファレンス

### `listEncodings(): string[]`

利用可能なエンコーディング名一覧（9 種類）。

### `getEncoding(name: string): Encoding`

エンコーディング名でトークナイザを取得。サポート:

- `cl100k_base` — GPT-4、GPT-3.5-turbo
- `o200k_base` — GPT-4o、GPT-4.1、o1、o3
- `p50k_base` — text-davinci-002/003
- `p50k_edit` — text-davinci-edit
- `r50k_base` — GPT-3（davinci、curie 等）
- `llama3` — Meta Llama 3/4
- `deepseek_v3` — DeepSeek V3/R1
- `qwen2` — Qwen 2/2.5/3
- `mistral_v3` — Mistral/Codestral/Pixtral

### `encodingForModel(model: string): Encoding`

モデル名でトークナイザを取得（例: `gpt-4o`、`llama-4-scout`、`deepseek-r1`、`qwen3-max`）。

### `modelToEncoding(model: string): string | null`

モデル名を対応するエンコーディング名にマップ（エンコーディング本体はロードしない）。

### `Encoding`

| メソッド / プロパティ | 型 | 説明 |
|-------------------|------|-------------|
| `encode(text)` | `Uint32Array` | テキスト → token id |
| `encodeWithSpecialTokens(text)` | `Uint32Array` | 特殊トークンを認識しながらエンコード |
| `decode(tokens)` | `string` | token id → テキスト |
| `count(text)` | `number` | トークン数のみ数える（`encode().length` より高速） |
| `countWithSpecialTokens(text)` | `number` | 特殊トークンを認識しながらカウント |
| `name` | `string` | エンコーディング名（getter） |
| `vocabSize` | `number` | 通常トークンの語彙サイズ |
| `numSpecialTokens` | `number` | 特殊トークン数 |
| `free()` | `void` | WASM メモリを解放 |

### `estimateCost(modelId, inputTokens, outputTokens): number`

USD で API コストを見積もり。7 プロバイダ 57 モデル対応。

### `getModelInfo(modelId): ModelInfo`

完全型付きのモデルメタデータを取得。

### `allModels(): ModelInfo[]`

サポート 57 モデルすべてを価格情報付きで列挙。

### `modelsByProvider(provider): ModelInfo[]`

プロバイダでモデルを絞り込み: `"OpenAI"`、`"Anthropic"`、`"Google"`、`"Meta"`、`"DeepSeek"`、`"Alibaba"`、`"Mistral"`。

### `ModelInfo`

| フィールド | 型 | 説明 |
|----------|------|-------------|
| `id` | `string` | モデル識別子 |
| `provider` | `string` | プロバイダ名 |
| `inputPer1m` | `number` | 入力 100 万 token あたりのコスト（USD） |
| `outputPer1m` | `number` | 出力 100 万 token あたりのコスト |
| `cachedInputPer1m` | `number \| undefined` | キャッシュ命中時の入力コスト |
| `contextWindow` | `number` | 最大コンテキストウィンドウ（token） |
| `maxOutput` | `number` | 最大出力 token |

## 対応モデル（価格情報付き）

| プロバイダ | モデル |
|----------|--------|
| OpenAI | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, o1, o1-mini, o1-pro, o3, o3-pro, o3-mini, o4-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002 |
| Anthropic | claude-opus-4.6, claude-sonnet-4.6, claude-haiku-4.5, claude-opus-4.5, claude-sonnet-4.5, claude-opus-4, claude-sonnet-4, claude-3.5-haiku, claude-3.5-sonnet, claude-3-opus, claude-3-haiku |
| Google | gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash, text-embedding-004 |
| Meta | llama-4-scout, llama-4-maverick, llama-3.1-405b, llama-3.1-70b, llama-3.1-8b, llama-3.3-70b |
| DeepSeek | deepseek-v3, deepseek-r1 |
| Alibaba | qwen3-max, qwen3-plus, qwen3-coder, qwen3-8b, qwen2.5-72b, qwen2.5-32b, qwen2.5-7b |
| Mistral | mistral-large, mistral-medium, mistral-small, mistral-nemo, codestral, pixtral-large, mixtral-8x7b |

## 注意事項

### 初期化

他の API を呼ぶ前に `await init()` を 1 回呼ぶ必要があります。WASM モジュールのロードとコンパイルを行い、以降の呼び出しは no-op になります。

### メモリ管理

`Encoding` インスタンスはグローバルにキャッシュされたデータへの参照を持つ軽量なものです。`.free()` を呼ぶと JS ラッパのみ解放され — 裏のエンコーディングデータは再利用のため引き続きキャッシュに残ります。短命なスクリプトでは `.free()` は省略可能ですが、長時間動くアプリでは使い終わり時に呼んでください。

## Demo

完全な Vite + React デモアプリ: [`examples/react-app`](../tiktoken/examples/react-app/)。

## ライセンス

[MIT](LICENSE)
