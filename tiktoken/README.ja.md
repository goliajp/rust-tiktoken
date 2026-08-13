# tiktoken

[![Crates.io](https://img.shields.io/crates/v/tiktoken?style=flat-square&logo=rust)](https://crates.io/crates/tiktoken)
[![docs.rs](https://img.shields.io/docsrs/tiktoken?style=flat-square&logo=docs.rs)](https://docs.rs/tiktoken)
[![License](https://img.shields.io/crates/l/tiktoken?style=flat-square)](#ライセンス)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square&logo=rust)](Cargo.toml)
[![Downloads](https://img.shields.io/crates/d/tiktoken?style=flat-square)](https://crates.io/crates/tiktoken)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

最速の Rust BPE トークナイザ — ネイティブで tiktoken-rs の 5〜49 倍（日本語・中国語の文章で 15〜17 倍）、ブラウザ内（wasm）で gpt-tokenizer の 2〜4 倍。手書きスキャナが ASCII と CJK の両方を扱い、語彙はキー長で階層化、繰り返し断片は丸ごとメモ化。OpenAI [tiktoken](https://github.com/openai/tiktoken) 互換で、**主要な全 LLM トークナイザ**をサポート — OpenAI、Llama 3、DeepSeek、Qwen、Mistral、Kimi、GLM、MiniMax。

## 特徴

- **マルチプロバイダ**: 8 社 17 エンコーディング（OpenAI、Meta、DeepSeek、Alibaba、Mistral、Moonshot、Zhipu、MiniMax）
- **高速**: 手書きスキャナが ASCII と CJK を処理（正規表現をバイパス）、キー長で階層化した語彙、断片の丸ごとメモ化、ハイブリッド BPE マージ
- **並列エンコード**: 大規模テキスト用のオプション rayon マルチスレッドエンコード
- **料金見積もり**: 11 プロバイダ 116 モデルのコスト推定
- **コンパクト**: 17 語彙で合計 5.1 MB を埋め込み、語彙ごとにオプトアウト可能 — cl100k のみのビルドなら 373 KB
- **ゼロアロケーションカウント**: `count()` パスはトークンベクタを割り当てません

## パフォーマンス

計測前に各実装のトークン出力が完全に一致することを確認し、各数値は 1 パス、
ウォームアップ後 9 回の中央値です。コーパスはすべてのハーネスでバイト単位に
同一（`bench-compare/`・`benches/bench_python.py`・`web/bench/`）。

#### ネイティブ — Apple M4 Mac mini・シングルスレッド・`encode`

`cargo run --release -p bench-compare`

| コーパス | Python tiktoken 0.12 | tiktoken-rs 0.9 | **tiktoken** | vs rs | vs Python |
|---|---|---|---|---|---|
| 短文 (13 B) | 1.6 µs | 1,081 ns | **33 ns** | **33x** | **48x** |
| 中程度 (900 B) | 31.9 µs | 52.2 µs | **1.1 µs** | **47x** | **29x** |
| 英語の文章 (45 KB) | 1,500 µs | 2,498 µs | **51.5 µs** | **49x** | **29x** |
| 中国語の文章 (4.3 KB) | 119.8 µs | 134.7 µs | **8.1 µs** | **17x** | **15x** |
| 日本語の文章 (4.6 KB) | 131.0 µs | 144.6 µs | **8.6 µs** | **17x** | **15x** |
| CJK 混在 ×50 (4.5 KB) | 138.9 µs | 160.3 µs | **15.2 µs** | **11x** | **9.2x** |
| 敵対的 CJK・繰り返しなし (3.9 KB) | 131.7 µs | 141.2 µs | **25.9 µs** | **5.5x** | **5.1x** |
| コード (3.9 KB) | 263.7 µs | 317.7 µs | **11.1 µs** | **29x** | **24x** |

o200k_base も同じ比率です（tiktoken-rs の 5〜48 倍）。`count()` は id ベクタを
一切確保しないため、`encode` よりさらに 5〜15% 高速です。

#### ブラウザ内 — Mac Studio (M4 Max)・Chromium

`web/` で `npm run bench` — 本クレートの wasm ビルドを、主要な 2 つの
JavaScript トークナイザーと比較。

| コーパス | gpt-tokenizer 3.4 | js-tiktoken 1.0 | **tiktoken (wasm)** |
|---|---|---|---|
| 中国語の文章 (4.3 KB) | 36.8 µs | 8,029 µs | **13.4 µs** |
| 日本語の文章 (4.6 KB) | 27.4 µs | 15,862 µs | **13.5 µs** |
| CJK 混在 ×50 (4.5 KB) | 41.2 µs | 4,665 µs | **24.2 µs** |
| 敵対的 CJK・繰り返しなし (3.9 KB) | 49.6 µs | 3,832 µs | **40.3 µs** |
| 英語の文章 (45 KB) | 478 µs | 7,010 µs | **112.5 µs** |
| コード (3.9 KB) | 76.0 µs | 916 µs | **19.5 µs** |

敵対的コーパスは断片の繰り返しを一切含まず、各実装のメモ化を無効にします —
それが下限であり、下限でも優位は変わりません。

<details>
<summary>なぜ速いのか</summary>

| | tiktoken | tiktoken-rs | Python tiktoken |
|---|---|---|---|
| 前処理分割 | 手書きスキャナが ASCII と CJK（漢字/かな/ハングル・全角形）の両方を処理；正規表現は判定基準と稀な形状のフォールバック | 常に正規表現 | 常に正規表現 |
| 正規表現エンジン（フォールバック） | `regex`（DFA・線形時間） | `fancy-regex`（バックトラック） | `regex`＋PyO3/FFI オーバーヘッド |
| 語彙引き | キー長で階層化：1〜2 バイトは直接表、3〜8 バイトはインラインスロットの開放アドレス法、それ以上はタグ付き arena スロット | `HashMap<Vec<u8>>`（20 万回のアロケーション） | PyO3 背後の Rust `HashMap` |
| 繰り返し断片 | スレッドローカルの直接マップ・メモ化（バイトキー） | なし | なし |
| BPE マージ | ハイブリッド：スタック線形走査（短い断片）＋ヒープ（長い断片） | O(n*m) 線形走査 | O(n*m) 線形走査 |
| アロケーションなしの `count()` | あり | なし | なし |

スキャナはプロパティテスト（毎回数十万件のランダム入力）で正規表現の語義に
固定され、各エンコーディングは 44,518 件の差分フィクスチャでベンダー自身の
トークナイザーに固定されています。

ベンチマークのソース：[`benches/`](benches/)・[`../bench-compare/`](../bench-compare/)。
`cargo bench` / `cargo run -p bench-compare` で再現できます。

</details>

## インストール

```toml
[dependencies]
tiktoken = "3.8"

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

`データ` 列は、そのエンコーディングの機能がバイナリに加える語彙バイト数です。同じデータファイルを共有するエンコーディングは二度目のコストがかからず、`+ ベース` の 3 つはランク整列した拡張で末尾のみを保持します。

| エンコーディング | プロバイダ | Feature | データ | 対応モデル |
|---|---|---|---|---|
| `o200k_base` | OpenAI | `vocab-o200k_base` | 815 KB | GPT-4o, GPT-4.1, GPT-4.5, GPT-5–5.6 (incl. Sol/Terra/Luna), o1, o3, o4-mini |
| `o200k_harmony` | OpenAI | `vocab-o200k_base` | — | gpt-oss（harmony チャットフォーマット） |
| `cl100k_base` | OpenAI | `vocab-cl100k_base` | 373 KB | GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, text-embedding-*, davinci-002, babbage-002 |
| `p50k_base` | OpenAI | `vocab-p50k_base` | 55 B + ベース | text-davinci-002/003, code-davinci-*, code-cushman-* |
| `p50k_edit` | OpenAI | `vocab-p50k_base` | — | text-davinci-edit-*, code-davinci-edit-* |
| `r50k_base` | OpenAI | `vocab-r50k_base` | 182 KB | GPT-3 世代: davinci, curie, babbage, ada |
| `gpt2` | OpenAI | `vocab-r50k_base` | — | GPT-2（`r50k_base` のエイリアス） |
| `llama3` | Meta | `vocab-llama3` | 111 KB + ベース | Llama 3, 3.1, 3.2, 3.3, 4 |
| `deepseek_v3` | DeepSeek | `vocab-deepseek_v3` | 514 KB | DeepSeek V3, R1 |
| `deepseek_v4` | DeepSeek | `vocab-deepseek_v3` | — | DeepSeek V4 Pro / Flash（V3 語彙 + V4 特殊トークン） |
| `qwen2` | Alibaba | `vocab-qwen2` | 564 KB | Qwen 2.5, Qwen 3 |
| `mistral_v3` | Mistral | `vocab-mistral_v3` | 525 KB | Mistral, Mixtral（Tekken トークナイザ） |
| `kimi_k2` | Moonshot | `vocab-kimi_k2` | 659 KB | Kimi K2 / K2.5 / K2.6 |
| `kimi_k3` | Moonshot | `vocab-kimi_k2` | — | Kimi K3（K2 語彙 + K3 特殊トークン） |
| `glm4` | Zhipu | `vocab-glm4` | 578 KB | GLM-4.5 / 4.6 / 4.7 |
| `glm5` | Zhipu | `vocab-glm5` | 6 KB + ベース | GLM-5 / 5.2 |
| `minimax_m2` | MiniMax | `vocab-minimax_m2` | 822 KB | MiniMax M2 / M2.1 / M2.5 / M2.7 |

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

// 各社 API が実際に受け付ける表記も解決します — 本表はドット、API はハイフン、
// リリース日サフィックス、Bedrock / Vertex の装飾
let r = pricing::resolve_model("us.anthropic.claude-opus-5").unwrap();
assert_eq!(r.model.id, "claude-opus-5");
assert!(matches!(r.matched, pricing::Match::Normalized { .. }));

// `estimate_cost` は同じ解決を通ります。`get_model` は完全一致のままです
assert!(pricing::estimate_cost("claude-haiku-4-5-20251001", 1_000, 1_000).is_some());
```

OpenAI、Anthropic、Google、Meta、DeepSeek、Alibaba、Mistral、Moonshot、Zhipu、
MiniMax、Voyage の 116 モデルに対応。

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
