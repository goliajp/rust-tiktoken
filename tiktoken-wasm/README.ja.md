# @goliapkg/tiktoken-wasm

[![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm?style=flat-square&logo=npm)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)
[![License](https://img.shields.io/npm/l/@goliapkg/tiktoken-wasm?style=flat-square)](#ライセンス)

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
- `tiktoken_wasm_bg.wasm` — コンパイル済み WASM バイナリ（8.96 MB、gzip 後 8.22 MB、全 20 語彙を含む）
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
// ["cl100k_base", "o200k_base", ..., "minimax_m2"]

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
const encName = modelToEncoding('llama-4-scout')  // "llama4"

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

利用可能なエンコーディング名一覧（11 種類）。

### `getEncoding(name: string): Encoding`

エンコーディング名でトークナイザを取得。サポート:

- `cl100k_base` — GPT-4、GPT-3.5-turbo
- `o200k_base` — GPT-4o、GPT-4.1、GPT-5–5.6、GPT-6（公式未確認）、o1、o3、o4-mini
- `o200k_harmony` — gpt-oss（harmony チャット形式）
- `p50k_base` — text-davinci-002/003
- `p50k_edit` — text-davinci-edit
- `r50k_base` — GPT-3（davinci、curie 等）
- `gpt2` — GPT-2（`r50k_base` のエイリアス）
- `llama3` — Meta Llama 3.x
- `llama4` — Meta Llama 4、Muse
- `deepseek_v3` — DeepSeek V3/R1
- `qwen2` — Qwen 2/2.5/3
- `qwen3_5` — Qwen 3.5〜3.8、Qwen 4（公式未確認）
- `mistral_v3` — Mistral/Codestral/Pixtral（Tekken）
- `mistral_v15` — Mistral Large 4（Tekken v15）
- `deepseek_v4` — DeepSeek V4 Pro/Flash、V4.1 Flash
- `kimi_k2` — Kimi K2/K2.5/K2.6
- `kimi_k3` — Kimi K3/K3.1
- `glm4` — GLM-4.5/4.6/4.7
- `glm5` — GLM-5/5.1/5.2/5.3
- `minimax_m2` — MiniMax M2・M3 ファミリー

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

USD で API コストを見積もり。11 プロバイダ 153 モデル対応。

### `getModelInfo(modelId): ModelInfo`

完全型付きのモデルメタデータを取得。

id は `estimateCost` と同じ方法で解決されるため、各社 API の表記
(`claude-haiku-4-5`、`anthropic.claude-opus-5`)でも該当モデルを返します。
返る `id` は本表自身の表記です。

### `allModels(): ModelInfo[]`

サポート 153 モデルすべてを価格情報付きで列挙。

### `modelsByProvider(provider): ModelInfo[]`

プロバイダでモデルを絞り込み: `"OpenAI"`、`"Anthropic"`、`"Google"`、`"Meta"`、`"DeepSeek"`、`"Alibaba"`、`"Mistral"`、`"Moonshot"`、`"Zhipu"`、`"MiniMax"`、`"Voyage"`。

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

`estimateCost` / `getModelInfo` / `allModels` は **11 プロバイダの 153 モデル**（2026-11 時点の価格）を扱います。プロバイダごと、新しい順：

| プロバイダ | モデル数 | 最新のエントリ |
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
| Voyage | 8 | voyage-4-large/4/4-lite/code-4, voyage-3-large, voyage-3.5(-lite), voyage-code-3（埋め込みモデル） |

完全な一覧はコードが正です：実行時に `allModels()` を呼んでください。

## 注意事項

### 初期化

他の API を呼ぶ前に `await init()` を 1 回呼ぶ必要があります。WASM モジュールのロードとコンパイルを行い、以降の呼び出しは no-op になります。

### メモリ管理

`Encoding` インスタンスはグローバルにキャッシュされたデータへの参照を持つ軽量なものです。`.free()` を呼ぶと JS ラッパのみ解放され — 裏のエンコーディングデータは再利用のため引き続きキャッシュに残ります。短命なスクリプトでは `.free()` は省略可能ですが、長時間動くアプリでは使い終わり時に呼んでください。

## Demo

完全な Vite + React デモアプリ: [`examples/react-app`](../tiktoken/examples/react-app/)。

<!-- ECOSYSTEM BEGIN (generated — edit ecosystem.toml, not this block) -->

## エコシステム

[tiktoken](https://crates.io/crates/tiktoken) · **@goliapkg/tiktoken-wasm** · [instructors](https://crates.io/crates/instructors) · [chunkedrs](https://crates.io/crates/chunkedrs) · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## ライセンス

コードは [MIT](LICENSE-MIT) または [Apache-2.0](LICENSE-APACHE) のいずれか、あなたの選択でライセンスされます。

組み込みの語彙はこれに含まれず、それぞれ公開元のライセンスに従います。[LICENSE-3RD-PARTY](LICENSE-3RD-PARTY) に各語彙の出典とライセンス、[`licenses/`](licenses/) に全文、[NOTICE](NOTICE) に各ライセンスが求める表示をまとめています。一部のライセンスには利用条件があります：Llama のライセンス（月間アクティブユーザー 7 億超の製品は Meta への別途ライセンス申請が必要）、`mistral_v15` の Mistral ライセンス（Model as a Service 事業者は商用ライセンスが必要）、一定規模を超えた場合の Kimi・MiniMax の表示条項、DeepSeek の利用制限条項。語彙の `vocab-*` フィーチャを有効にしないビルドには、その語彙は含まれません。

Built with Llama.
