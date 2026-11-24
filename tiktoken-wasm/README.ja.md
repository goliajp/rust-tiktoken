# @goliapkg/tiktoken-wasm

[![npm](https://img.shields.io/npm/v/@goliapkg/tiktoken-wasm?style=flat-square&logo=npm)](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm)
[![License](https://img.shields.io/npm/l/@goliapkg/tiktoken-wasm?style=flat-square)](#ライセンス)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

BPE トークナイザー [tiktoken](https://crates.io/crates/tiktoken) の WebAssembly バインディングです。各社のモデルに合わせた token 分割を、ブラウザや Node.js で直接、ネイティブ版に近い速度で実行できます。

## インストール

```bash
npm install @goliapkg/tiktoken-wasm
```

このモジュールは WebAssembly SIMD（`simd128`）を使ってビルドしています。対応ブラウザは Chrome 91 以降、Firefox 89 以降、Safari 16.4 以降です。それより古いバージョンではモジュールをコンパイルできません。

## ソースからビルド

```bash
# wasm-pack が必要: cargo install wasm-pack
cd tiktoken-wasm
wasm-pack build --target web --release --scope goliapkg
```

ビルドすると、そのまま npm に公開できるパッケージが `pkg/` に生成されます。内容は次のとおりです。

- `tiktoken_wasm.js`：wasm ローダーを含む ES モジュール
- `tiktoken_wasm_bg.wasm`：全 20 種類の語彙を含むコンパイル済みの wasm バイナリ（8.65 MB、gzip 圧縮後は 8.01 MB）
- `tiktoken_wasm.d.ts`：TypeScript の型定義
- `snippets/`：テキストと token ID を wasm メモリに読み書きする小さな JavaScript 関数

自分でビルドする場合は、`--no-default-features --features vocab-o200k_base` で語彙を絞ると、バイナリは 1.64 MB、gzip 圧縮後は 1.00 MB になります。同じ方法で、`tiktoken` から引き継いだほかの `vocab-*` feature や提供元単位の feature も指定できます。

## 使い方

### ES モジュール（ブラウザ、Vite、webpack）

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

// モデル名から取得 — OpenAI、Meta、DeepSeek、Qwen、Mistral、Moonshot、Zhipu、MiniMax をサポート
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

### バンドラーの設定

Vite では、`vite.config.ts` に次のプラグインを追加します。

```typescript
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'

export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
})
```

webpack 5 では、`webpack.config.js` の実験的機能で wasm を有効にします。

```javascript
module.exports = {
  experiments: {
    asyncWebAssembly: true,
  },
}
```

Next.js では、`next.config.js` に次の設定を追加します。

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

利用できるエンコーディング 20 種類の名前をすべて返します。

### `getEncoding(name: string): Encoding`

指定したエンコーディング名に対応するトークナイザーを返します。利用できる名前と対応モデルは次のとおりです。

- `cl100k_base`：GPT-4、GPT-3.5-turbo
- `o200k_base`：GPT-4o、GPT-4.1、GPT-5〜5.6、GPT-6（公式未確認）、o1、o3、o4-mini
- `o200k_harmony`：gpt-oss（harmony チャット形式）
- `p50k_base`：text-davinci-002/003
- `p50k_edit`：text-davinci-edit
- `r50k_base`：GPT-3（davinci、curie など）
- `gpt2`：GPT-2（`r50k_base` の別名）
- `llama3`：Meta Llama 3.x
- `llama4`：Meta Llama 4、Muse
- `deepseek_v3`：DeepSeek V3/R1
- `qwen2`：Qwen 2/2.5/3
- `qwen3_5`：Qwen 3.5〜3.8、Qwen 4（公式未確認）
- `mistral_v3`：Mistral/Codestral/Pixtral（Tekken）
- `mistral_v15`：Mistral Large 4（Tekken v15）
- `deepseek_v4`：DeepSeek V4 Pro/Flash、V4.1 Flash
- `kimi_k2`：Kimi K2/K2.5/K2.6
- `kimi_k3`：Kimi K3/K3.1
- `glm4`：GLM-4.5/4.6/4.7
- `glm5`：GLM-5/5.1/5.2/5.3
- `minimax_m2`：MiniMax M2 系と M3 系

### `encodingForModel(model: string): Encoding`

モデル名（`gpt-4o`、`llama-4-scout`、`deepseek-r1`、`qwen3-max` など）に対応するトークナイザーを返します。

### `modelToEncoding(model: string): string | null`

モデル名からエンコーディング名を取得します。エンコーディング本体は読み込みません。

### `Encoding`

| メソッド / プロパティ | 型 | 説明 |
|-------------------|------|-------------|
| `encode(text)` | `Uint32Array` | テキストを token ID 列に変換 |
| `encodeWithSpecialTokens(text)` | `Uint32Array` | 特殊 tokenを認識しながらエンコード |
| `decode(tokens)` | `string` | token ID 列をテキストに戻す |
| `count(text)` | `number` | token 数を数える（`encode().length` より高速） |
| `countWithSpecialTokens(text)` | `number` | 特殊 tokenを認識して token 数を数える |
| `name` | `string` | エンコーディング名（getter） |
| `vocabSize` | `number` | 語彙に含まれる通常の token の数 |
| `numSpecialTokens` | `number` | 特殊 token の数 |
| `free()` | `void` | wasm メモリを解放 |

### `estimateCost(modelId, inputTokens, outputTokens): number`

入力と出力の token 数から API の利用料金を米ドルで見積もります。11 社の 153 モデルに対応しています。

### `getModelInfo(modelId): ModelInfo`

TypeScript ですべての項目に型が付いたモデル情報を返します。

モデル ID は `estimateCost` と同じ規則で検索します。各社の API で使う表記（`claude-haiku-4-5`、`anthropic.claude-opus-5` など）を指定しても、該当するモデルの情報が返ります。戻り値の `id` には料金表での表記を使います。

### `allModels(): ModelInfo[]`

対応する全 153 モデルを料金情報とともに返します。

### `modelsByProvider(provider): ModelInfo[]`

指定した提供元のモデルだけを返します。指定できる値は `"OpenAI"`、`"Anthropic"`、`"Google"`、`"Meta"`、`"DeepSeek"`、`"Alibaba"`、`"Mistral"`、`"Moonshot"`、`"Zhipu"`、`"MiniMax"`、`"Voyage"` です。

### `ModelInfo`

| フィールド | 型 | 説明 |
|----------|------|-------------|
| `id` | `string` | モデル識別子 |
| `provider` | `string` | プロバイダ名 |
| `inputPer1m` | `number` | 入力 100 万 token あたりの料金（USD） |
| `outputPer1m` | `number` | 出力 100 万 token あたりの料金（USD） |
| `cachedInputPer1m` | `number \| undefined` | キャッシュ済み入力 100 万 token あたりの料金 |
| `contextWindow` | `number` | コンテキスト長の上限（token 数） |
| `maxOutput` | `number` | 出力 token 数の上限 |

## 対応モデル（価格情報付き）

`estimateCost`、`getModelInfo`、`allModels` は 11 社の 153 モデルに対応しています。料金は 2026 年 11 月時点のものです。下表では提供元ごとにモデルを新しい順に掲載しています。

| プロバイダ | モデル数 | 最近のモデル |
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
| Voyage | 8 | voyage-4-large/4/4-lite/code-4、voyage-3-large、voyage-3.5(-lite)、voyage-code-3（埋め込みモデル） |

対応モデルの正式な一覧はコードで定義しています。全件を確認するには、実行時に `allModels()` を呼ぶか、[docs.rs](https://docs.rs/tiktoken/latest/tiktoken/pricing/index.html) を参照してください。Anthropic と Google は照合先となるトークナイザーを公開していないため、料金情報のみ提供しています。ほかの 8 社については token 分割にも対応しています。

## 注意事項

### 初期化

ほかの API を使う前に `await init()` を 1 回呼び出してください。初回の呼び出しで wasm モジュールを読み込み、コンパイルします。2 回目以降の呼び出しでは何も実行しません。

### メモリ管理

`Encoding` インスタンスは、グローバルキャッシュ内のデータを参照する軽量なオブジェクトです。`.free()` を呼ぶと JavaScript 側のラッパーを解放します。エンコーディングのデータは再利用できるようキャッシュに残ります。すぐに終了するスクリプトでは `.free()` を省略できます。長時間動作するアプリケーションでは、インスタンスを使い終えたら呼び出してください。

## デモ

Vite と React を使ったデモアプリ一式は [`examples/react-app`](../tiktoken/examples/react-app/) にあります。

<!-- ECOSYSTEM BEGIN (generated — edit ecosystem.toml, not this block) -->

## エコシステム

[tiktoken](https://crates.io/crates/tiktoken) · **@goliapkg/tiktoken-wasm** · [instructors](https://crates.io/crates/instructors) · [chunkedrs](https://crates.io/crates/chunkedrs) · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## ライセンス

コードのライセンスは [MIT](LICENSE-MIT) と [Apache-2.0](LICENSE-APACHE) のどちらかを選べます。

組み込みの語彙には、コードとは別に各公開元のライセンスが適用されます。各語彙の出典とライセンスは [LICENSE-3RD-PARTY](LICENSE-3RD-PARTY)、ライセンス全文は [`licenses/`](licenses/)、各ライセンスが求める帰属表示は [NOTICE](NOTICE) に掲載しています。

Llama のライセンスでは、月間アクティブユーザー数が 7 億人を超える製品には Meta からのライセンスが必要です。Mistral の `mistral_v15` を Model-as-a-Service 事業で利用する場合は商用ライセンスが必要です。Kimi と MiniMax では一定の規模を超えると表示義務が生じます。DeepSeek には用途に応じた制限があります。ビルド時に各語彙の `vocab-*` feature を無効にすると、その語彙は組み込まれません。

Built with Llama.
