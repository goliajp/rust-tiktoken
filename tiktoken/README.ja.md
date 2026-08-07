# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#ライセンス)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square&logo=rust)](Cargo.toml)
[![Downloads](https://img.shields.io/crates/d/tiktoken?style=flat-square)](https://crates.io/crates/tiktoken)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

最速の Rust BPE トークナイザ — ASCII テキストで tiktoken-rs より 15〜40 倍高速（CJK/Unicode では約 2 倍）。手書きの ASCII 高速パスによる。OpenAI [tiktoken](https://github.com/openai/tiktoken) 互換で、**主要な全 LLM トークナイザ**をサポート — OpenAI、Llama 3、DeepSeek、Qwen、Mistral。

## 特徴

- **マルチプロバイダ**: 5 社 11 エンコーディング（OpenAI、Meta、DeepSeek、Alibaba、Mistral）
- **高速**: 手書き ASCII 高速パス（正規表現をバイパス）、Arena ベースの語彙、ハイブリッド BPE マージ
- **並列エンコード**: 大規模テキスト用のオプション rayon マルチスレッドエンコード
- **料金見積もり**: 7 プロバイダ 94 モデルのコスト推定
- **コンパクト**: ruzstd 圧縮語彙データをコンパイル時に埋め込み
- **ゼロアロケーションカウント**: `count()` パスはトークンベクタを割り当てません

## パフォーマンス

Apple M4 Mac mini、シングルスレッドで測定。3 実装のトークン出力が完全一致することを検証済み。

#### cl100k_base encode

| 入力 | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken** | vs tiktoken-rs | vs Python |
|---|---|---|---|---|---|
| 短文 (13 B) | 1,700 ns | 1,248 ns | **43 ns** | **29x** | **40x** |
| 中文 (900 B) | 32.2 us | 53.8 us | **1.5 us** | **35x** | **21x** |
| 長文 (45 KB) | 1,500 us | 2,611 us | **74 us** | **35x** | **20x** |
| Unicode (4.5 KB) | 141 us | 164 us | **91 us** | **1.8x** | **1.6x** |
| コード (3.9 KB) | 247 us | 264 us | **17 us** | **16x** | **15x** |

#### o200k_base encode

| 入力 | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken** | vs tiktoken-rs | vs Python |
|---|---|---|---|---|---|
| 短文 (13 B) | 1,600 ns | 1,051 ns | **40 ns** | **26x** | **40x** |
| 中文 (900 B) | 58.3 us | 56.2 us | **1.5 us** | **37x** | **39x** |
| 長文 (45 KB) | 2,900 us | 2,799 us | **73 us** | **38x** | **40x** |
| Unicode (4.5 KB) | 204 us | 187 us | **95 us** | **2.0x** | **2.2x** |
| コード (3.9 KB) | 332 us | 253 us | **16 us** | **16x** | **21x** |

<details>
<summary>なぜ速いのか？</summary>

| | tiktoken | tiktoken-rs | Python tiktoken |
|---|---|---|---|
| 語彙ストレージ | Arena ベース（単一アロケーション、キャッシュフレンドリー） | `HashMap<Vec<u8>>`（20 万回アロケーション） | PyO3 背後の Rust `HashMap` |
| 事前トークン化（ASCII） | 手書き ASCII 高速パス、正規表現をスキップ | 常に正規表現を実行 | 常に正規表現を実行 |
| 正規表現エンジン（フォールバック） | `regex`（DFA、線形時間） | `fancy-regex`（バックトラッキング） | `regex`（PyO3 + FFI オーバーヘッド経由） |
| ハッシュマップ | カスタムオープンアドレス + `FxHash` | `rustc-hash` v1 | 標準 `HashMap` |
| BPE マージ | ハイブリッド: スタック線形スキャン（短い断片）+ ヒープ（長い断片） | O(n*m) 線形スキャン | O(n*m) 線形スキャン |
| ゼロアロケーション `count()` | あり | なし | なし |

ベンチマークソース：[`benches/`](benches/)。`cargo bench` で再現可能。

</details>

## インストール

```toml
[dependencies]
tiktoken = "3"

# オプション: 大規模テキストのマルチスレッドエンコード
tiktoken = { version = "3", features = ["parallel"] }
```

## クイックスタート

```rust
// エンコーディング名で取得
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode("hello world");
let text = enc.decode_to_string(&tokens).unwrap();
assert_eq!(text, "hello world");

// モデル名で取得 — 全プロバイダ対応
let enc = tiktoken::encoding_for_model("gpt-4o").unwrap();
let count = enc.count("hello world"); // ゼロアロケーション高速パス

let enc = tiktoken::encoding_for_model("llama-3.1-70b").unwrap();
let enc = tiktoken::encoding_for_model("deepseek-v3").unwrap();
let enc = tiktoken::encoding_for_model("qwen2.5-72b").unwrap();
```

## 対応エンコーディング

| エンコーディング | プロバイダ | 対応モデル |
|---|---|---|
| `o200k_base` | OpenAI | GPT-4o, GPT-4.1, GPT-4.5, GPT-5–5.6 (incl. Sol/Terra/Luna), o1, o3, o4-mini |
| `o200k_harmony` | OpenAI | gpt-oss（harmony チャットフォーマット） |
| `cl100k_base` | OpenAI | GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, text-embedding-*, davinci-002, babbage-002 |
| `p50k_base` | OpenAI | text-davinci-002/003, code-davinci-*, code-cushman-* |
| `p50k_edit` | OpenAI | text-davinci-edit-*, code-davinci-edit-* |
| `r50k_base` | OpenAI | GPT-3 世代: davinci, curie, babbage, ada |
| `gpt2` | OpenAI | GPT-2（`r50k_base` のエイリアス） |
| `llama3` | Meta | Llama 3, 3.1, 3.2, 3.3, 4 |
| `deepseek_v3` | DeepSeek | DeepSeek V3, R1 |
| `qwen2` | Alibaba | Qwen 2.5, Qwen 3 |
| `mistral_v3` | Mistral | Mistral, Mixtral（Tekken トークナイザ） |

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

// 特殊トークン認識付きカウント
let count = enc.count_with_special_tokens("hello<|endoftext|>world");
```

### 並列エンコード

```rust
// `parallel` feature が必要
let enc = tiktoken::get_encoding("cl100k_base").unwrap();
let tokens = enc.encode_parallel("...非常に長いテキスト...");
// 出力は完全に同一、>= 4KB テキストで rayon を使用
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
let models = pricing::models_by_provider(pricing::Provider::DeepSeek);
```

OpenAI、Anthropic、Google、Meta、DeepSeek、Alibaba、Mistral の 94 モデルに対応。

## WebAssembly

npm パッケージ [`@goliapkg/tiktoken-wasm`](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) として公開中。ブラウザと Node.js で利用可能。

```bash
npm install @goliapkg/tiktoken-wasm
```

```typescript
import init, { getEncoding, encodingForModel, estimateCost } from '@goliapkg/tiktoken-wasm'

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

<!-- ECOSYSTEM BEGIN (generated — edit ecosystem.toml, not this block) -->

## エコシステム

**tiktoken** · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · [chunkedrs](https://crates.io/crates/chunkedrs) · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## ライセンス

[MIT](LICENSE-MIT) または [Apache-2.0](LICENSE-APACHE) のいずれか、あなたの選択でライセンスされます。

サードパーティ語彙データのライセンス: [LICENSE-3RD-PARTY](LICENSE-3RD-PARTY) を参照。
