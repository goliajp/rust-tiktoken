# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen?style=flat-square)](src/)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-blue?style=flat-square&logo=rust)](Cargo.toml)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

OpenAI の [tiktoken](https://github.com/openai/tiktoken) 互換の最速 Rust BPE トークナイザ。**tiktoken-rs より 7〜10 倍高速**、**Python tiktoken より 4〜15 倍高速**。

## パフォーマンス

Apple M4 Mac mini、シングルスレッドで測定。3 実装のトークン出力が完全一致することを検証済み。

#### cl100k_base encode

| 入力 | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken 2.1** | vs tiktoken-rs | vs Python |
|---|---|---|---|---|---|
| 短文 (13 B) | 1,700 ns | 1,248 ns | **118 ns** | **10.5x** | **14x** |
| 中文 (900 B) | 32.2 µs | 53.8 µs | **7.3 µs** | **7.3x** | **4.4x** |
| 長文 (45 KB) | 1,500 µs | 2,611 µs | **373 µs** | **7.0x** | **4.0x** |
| Unicode (4.5 KB) | 141 µs | 164 µs | **97 µs** | **1.7x** | **1.5x** |
| コード (3.9 KB) | 247 µs | 264 µs | **34 µs** | **7.7x** | **7.3x** |

#### o200k_base encode

| 入力 | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken 2.1** | vs tiktoken-rs | vs Python |
|---|---|---|---|---|---|
| 短文 (13 B) | 1,600 ns | 1,051 ns | **116 ns** | **9.1x** | **14x** |
| 中文 (900 B) | 58.3 µs | 56.2 µs | **7.3 µs** | **7.7x** | **8.0x** |
| 長文 (45 KB) | 2,900 µs | 2,799 µs | **374 µs** | **7.5x** | **7.8x** |
| Unicode (4.5 KB) | 204 µs | 187 µs | **104 µs** | **1.8x** | **2.0x** |
| コード (3.9 KB) | 332 µs | 253 µs | **33 µs** | **7.6x** | **10x** |

<details>
<summary>なぜ速いのか？</summary>

| | tiktoken | tiktoken-rs | Python tiktoken |
|---|---|---|---|
| 正規表現エンジン | `regex`（DFA、線形時間） | `fancy-regex`（バックトラッキング） | `regex`（PyO3 + FFI オーバーヘッド経由） |
| ハッシュマップ | `FxHashMap`（小キーに最適化） | `rustc-hash` v1 | PyO3 背後の Rust `HashMap` |
| BPE マージ | ランクキャッシュ（隣接 2 要素のみ再計算） | 標準実装 | 標準実装 |
| ゼロアロケーション `count()` | あり | なし | なし |
| ランタイム依存 | 4 crates | 16 crates | — |

ベンチマークソース：[`benches/`](benches/)。`cargo bench` で再現可能。

</details>

## インストール

```toml
[dependencies]
tiktoken = "2.1"
```

## クイックスタート

```rust
// エンコーディング名で取得
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode("hello world");
let text = enc.decode_to_string(&tokens).unwrap();
assert_eq!(text, "hello world");

// モデル名で取得
let enc = tiktoken::encoding_for_model("gpt-4o").unwrap();
let count = enc.count("hello world"); // ゼロアロケーション高速パス
```

## 対応エンコーディング

| エンコーディング | 対応モデル |
|---|---|
| `o200k_base` | GPT-4o, GPT-4o-mini, o1, o3, o4-mini |
| `cl100k_base` | GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, text-embedding-ada-002, text-embedding-3-* |
| `p50k_base` | text-davinci-002/003, code-davinci-*, code-cushman-* |
| `p50k_edit` | text-davinci-edit-*, code-davinci-edit-* |
| `r50k_base` | text-davinci-001, text-curie, text-babbage, text-ada, davinci, curie, babbage, ada |

## API

### エンコード / デコード

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();

let tokens = enc.encode("hello world");           // Vec<u32>
let text = enc.decode_to_string(&tokens).unwrap(); // String
let bytes = enc.decode(&tokens);                   // Vec<u8>
```

### 特殊トークン

```rust
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode_with_special_tokens("hello<|endoftext|>world");
// tokens に <|endoftext|> の特殊トークン ID が含まれます
```

### トークンカウント

```rust
let enc = tiktoken::get_encoding("o200k_base").unwrap();
let count = enc.count("素早い茶色の狐が怠けた犬を飛び越えた。");
// encode().len() より高速 — トークンベクタを割り当てません
```

### コスト見積もり

```rust
use tiktoken::pricing;

// 簡易見積もり
let cost = pricing::estimate_cost("gpt-4o", 1_000_000, 500_000).unwrap();

// プロンプトキャッシュ付き
let model = pricing::get_model("claude-opus-4").unwrap();
let cost = model.estimate_cost_with_cache(500_000, 500_000, 200_000);

// プロバイダ別のモデル一覧
let models = pricing::models_by_provider(pricing::Provider::Google);
```

OpenAI、Anthropic Claude、Google Gemini の 26 モデルに対応。

## WebAssembly

npm パッケージ [`@goliapkg/tokenrs-wasm`](https://www.npmjs.com/package/@goliapkg/tokenrs-wasm) として公開中。ブラウザと Node.js で利用可能。

```bash
npm install @goliapkg/tokenrs-wasm
```

```typescript
import init, { getEncoding, encodingForModel, estimateCost } from '@goliapkg/tokenrs-wasm'

await init()

const enc = getEncoding('cl100k_base')
const tokens = enc.encode('hello world')    // Uint32Array
const text = enc.decode(tokens)             // "hello world"
const count = enc.count('hello world')      // 2

enc.free() // 使用後に WASM メモリを解放
```

| バンドラ | プラグイン |
|---------|--------|
| Vite | [vite-plugin-wasm](https://www.npmjs.com/package/vite-plugin-wasm) + [vite-plugin-top-level-await](https://www.npmjs.com/package/vite-plugin-top-level-await) |
| webpack 5 | 組み込み `asyncWebAssembly` 実験機能 |
| Next.js | [next.config.js `webpack.experiments`](https://nextjs.org/docs/app/api-reference/next-config-js/webpack) |

完全なデモは [`examples/react-app`](examples/react-app/) を参照。

<details>
<summary>ソースからビルド</summary>

```bash
# wasm-pack が必要: cargo install wasm-pack
cd tiktoken-wasm
wasm-pack build --target web --release
```

</details>

## ライセンス

[MIT](LICENSE)
